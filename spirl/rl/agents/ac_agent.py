import torch
import os
import numpy as np

from spirl.rl.components.agent import BaseAgent
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from spirl.rl.utils.mpi import sync_networks


class ACAgent(BaseAgent):
    """Implements actor-critic agent. (does not implement update function, this should be handled by RL algo agent)"""
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)
        if self.policy.has_trainable_params:
            self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy, self._hp.policy_lr)

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy': None,     # policy class
            'policy_params': None,  # parameters for the policy class
            'policy_lr': 3e-4,  # learning rate for policy update
        })
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        # TODO implement non-sampling validation mode
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        if len(obs.shape) == 1:     # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None]))
            if 'dist' in policy_output:
                del policy_output['dist']
            return map2np(policy_output)
        return map2np(self.policy(obs))

    def _act_rand(self, obs):
        policy_output = self.policy.sample_rand(map2torch(obs, self.policy.device))
        if 'dist' in policy_output:
            del policy_output['dist']
        return map2np(policy_output)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        if self.policy.has_trainable_params:
            d['policy_opt'] = self.policy_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop('policy_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def reset(self):
        self.policy.reset()

    def sync_networks(self):
        if self.policy.has_trainable_params:
            sync_networks(self.policy)

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch


class SACAgent(ACAgent):
    """Implements SAC algorithm."""
    def __init__(self, config):
        ACAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)

        # build critics and target networks, copy weights of critics to target networks
        self.critics = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(2)])
        self.critic_targets = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(2)])
        [self._copy_to_target_network(target, source) for target, source in zip(self.critics, self.critic_targets)]

        # build optimizers for critics
        self.critic_opts = [self._get_optimizer(self._hp.optimizer, critic, self._hp.critic_lr) for critic in self.critics]

        # define entropy multiplier alpha
        self._log_alpha = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
        self.alpha_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha, self._hp.alpha_lr)
        self._target_entropy = self._hp.target_entropy if self._hp.target_entropy is not None \
                                        else -1 * self._hp.policy_params.action_dim

        # build replay buffer
        self.replay_buffer = self._hp.replay(self._hp.replay_params)

        self._update_steps = 0                # counts the number of alpha updates for optional variable schedules

    def _default_hparams(self):
        default_dict = ParamDict({
            'critic': None,           # critic class
            'critic_params': None,    # parameters for the critic class
            'replay': None,           # replay buffer class
            'replay_params': None,    # parameters for replay buffer
            'critic_lr': 3e-4,        # learning rate for critic update
            'alpha_lr': 3e-4,         # learning rate for alpha coefficient update
            'reward_scale': 1.0,      # SAC reward scale
            'clip_q_target': False,   # if True, clips Q target
            'target_entropy': None,   # target value for automatic entropy tuning, if None uses -action_dim
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        """Updates actor and critics."""
        # push experience batch into replay buffer
        self.add_experience(experience_batch)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            policy_output = self._run_policy(experience_batch.observation)

            # update alpha
            alpha_loss = self._update_alpha(experience_batch, policy_output)

            # compute policy loss
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)

            # compute target Q value
            with torch.no_grad():
                policy_output_next = self._run_policy(experience_batch.observation_next)
                value_next = self._compute_next_value(experience_batch, policy_output_next)
                q_target = experience_batch.reward * self._hp.reward_scale + \
                                (1 - experience_batch.done) * self._hp.discount_factor * value_next
                if self._hp.clip_q_target:
                    q_target = self._clip_q_target(q_target)
                q_target = q_target.detach()
                check_shape(q_target, [self._hp.batch_size])

            # compute critic loss
            critic_losses, qs = self._compute_critic_loss(experience_batch, q_target)

            # update critic networks
            [self._perform_update(critic_loss, critic_opt, critic)
                    for critic_loss, critic_opt, critic in zip(critic_losses, self.critic_opts, self.critics)]

            # update target networks
            [self._soft_update_target_network(critic_target, critic)
                    for critic_target, critic in zip(self.critic_targets, self.critics)]

            # update policy network on policy loss
            self._perform_update(policy_loss, self.policy_opt, self.policy)

            # logging
            info = AttrDict(    # losses
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
                critic_loss_1=critic_losses[0],
                critic_loss_2=critic_losses[1],
            )
            if self._update_steps % 100 == 0:
                info.update(AttrDict(       # gradient norms
                    policy_grad_norm=avg_grad_norm(self.policy),
                    critic_1_grad_norm=avg_grad_norm(self.critics[0]),
                    critic_2_grad_norm=avg_grad_norm(self.critics[1]),
                ))
            info.update(AttrDict(       # misc
                alpha=self.alpha,
                pi_log_prob=policy_output.log_prob.mean(),
                policy_entropy=policy_output.dist.entropy().mean(),
                q_target=q_target.mean(),
                q_1=qs[0].mean(),
                q_2=qs[1].mean(),
            ))
            info.update(self._aux_info(policy_output))
            info = map_dict(ten2ar, info)

        return info

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer."""
        if not experience_batch:
            return  # pass if experience_batch is empty
        self.replay_buffer.append(experience_batch)
        self._obs_normalizer.update(experience_batch.observation)

    def _sample_experience(self):
        return self.replay_buffer.sample(n_samples=self._hp.batch_size)

    def _normalize_batch(self, experience_batch):
        """Optionally apply observation normalization."""
        experience_batch.observation = self._obs_normalizer(experience_batch.observation)
        experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
        return experience_batch

    def _run_policy(self, obs):
        """Allows child classes to post-process policy outputs."""
        return self.policy(obs)

    def _update_alpha(self, experience_batch, policy_output):
        alpha_loss = self._compute_alpha_loss(policy_output)
        self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha)
        return alpha_loss

    def _compute_alpha_loss(self, policy_output):
        self._update_steps += 1
        return -1 * (self.alpha * (self._target_entropy + policy_output.log_prob).detach()).mean()

    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        next_val = (q_next - self.alpha * policy_output.log_prob[:, None])
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _compute_critic_loss(self, experience_batch, q_target):
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs

    def _compute_q_estimates(self, experience_batch):
        return [critic(experience_batch.observation, self._prep_action(experience_batch.action.detach())).q.squeeze(-1)
                    for critic in self.critics]     # no gradient propagation into policy here!

    def _prep_action(self, action):
        """Preprocessing of action in case of discrete action space."""
        if len(action.shape) == 1: action = action[:, None]  # unsqueeze for single-dim action spaces
        return action.float()

    def _clip_q_target(self, q_target):
        clip = 1 / (1 - self._hp.discount_factor)
        return torch.clamp(q_target, -clip, clip)

    def _aux_info(self, policy_output):
        """Optionally add auxiliary info about policy outputs etc."""
        return AttrDict()

    def sync_networks(self):
        super().sync_networks()
        [sync_networks(critic) for critic in self.critics]
        sync_networks(self._log_alpha)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        d['critic_opts'] = [o.state_dict() for o in self.critic_opts]
        d['alpha_opt'] = self.alpha_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        [o.load_state_dict(d) for o, d in zip(self.critic_opts, state_dict.pop('critic_opts'))]
        self.alpha_opt.load_state_dict(state_dict.pop('alpha_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def save_state(self, save_dir):
        """Saves compressed replay buffer to disk."""
        self.replay_buffer.save(os.path.join(save_dir, 'replay'))

    def load_state(self, save_dir):
        """Loads replay buffer from disk."""
        self.replay_buffer.load(os.path.join(save_dir, 'replay'))

    @property
    def alpha(self):
        return self._log_alpha().exp()
