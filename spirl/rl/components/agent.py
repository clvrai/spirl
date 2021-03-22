import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from functools import partial
from torch.optim import Adam, SGD

from spirl.utils.general_utils import ParamDict, get_clipped_optimizer, AttrDict, prefix_dict, map_dict, \
                                        nan_hook, np2obj, ConstantSchedule
from spirl.utils.pytorch_utils import RAdam, remove_grads, map2np, map2torch
from spirl.utils.vis_utils import add_caption_to_img, add_captions_to_seq
from spirl.rl.components.normalization import DummyNormalizer
from spirl.rl.components.policy import Policy
from spirl.components.checkpointer import CheckpointHandler
from spirl.rl.utils.mpi import sync_grads


class BaseAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)
        self.device = self._hp.device
        self._is_train = True           # indicates whether agent should sample in training mode
        self._rand_act_mode = False     # indicates whether agent should act randomly (for warmup collection)
        self._rollout_mode = False      # indicates whether agent is run in rollout mode (omit certain policy outputs)
        self._obs_normalizer = self._hp.obs_normalizer(self._hp.obs_normalizer_params)

    def _default_hparams(self):
        default_dict = ParamDict({
            'device': None,                         # pytorch device
            'discount_factor': 0.99,                # discount factor for RL update
            'optimizer': 'adam',                    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'gradient_clip': None,                  # max grad norm, if None no clipping
            'momentum': 0,                          # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,                       # beta1 param in Adam
            'update_iterations': 1,                 # number of iteration steps per one call to 'update(...)'
            'target_network_update_factor': 5e-3,   # percentage of new weights that are carried over
            'batch_size': 64,                       # size of the experience batch used for updates
            'obs_normalizer': DummyNormalizer,      # observation normalization class
            'obs_normalizer_params': {},            # parameters for optimization norm class
            'obs_norm_log_groups': {},              # (optional) dict defining separation of state space for obsNormLog
            'log_videos': True,                     # whether to log videos during logging
            'log_video_caption': False,             # whether to add captions to video
            'num_workers': None,                    # number of independent workers --> whether grads need sync
        })
        return default_dict

    def act(self, obs):
        """Returns policy output dict given observation (random action if self._rand_act_mode is set)."""
        if self._rand_act_mode:
            return self._act_rand(obs)
        else:
            return self._act(obs)

    def _act(self, obs):
        """Implements act method in child class."""
        raise NotImplementedError

    def _act_rand(self, obs):
        """Returns random action with proper dimension. Implemented in child class."""
        raise NotImplementedError

    def update(self, experience_batch):
        """Updates the policy given a batch of experience."""
        raise NotImplementedError

    def add_experience(self, experience_batch):
        """Provides interface for adding additional experience to agent replay, needs to be overwritten by child."""
        print("### This agent does not support additional experience! ###")

    def log_outputs(self, logging_stats, rollout_storage, logger, log_images, step):
        """Visualizes/logs all training outputs."""
        logger.log_scalar_dict(logging_stats, prefix='train' if self._is_train else 'val', step=step)

        if log_images:
            assert rollout_storage is not None      # need rollout data for image logging
            # log rollout videos with info captions
            if 'image' in rollout_storage and self._hp.log_videos:
                if self._hp.log_video_caption:
                    vids = [np.stack(add_captions_to_seq(rollout.image, np2obj(rollout.info))).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-logger.n_logged_samples:]]
                else:
                    vids = [np.stack(rollout.image).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-logger.n_logged_samples:]]
                logger.log_videos(vids, name="rollouts", step=step)
            self.visualize(logger, rollout_storage, step)

    def visualize(self, logger, rollout_storage, step):
        """Optionally allows to further visualize the internal state of agent (e.g. replay buffer etc.)"""
        pass

    def reset(self):
        """Can be used for any initializations of agent's state at beginning of episode."""
        pass

    def save_state(self, save_dir):
        """Provides interface to save any internal state variables (like replay buffers) to disk."""
        pass

    def load_state(self, save_dir):
        """Provides interface to load any internal state variables (like replay buffers) from disk."""
        pass

    def sync_networks(self):
        """Syncs network parameters across workers."""
        raise NotImplementedError

    def _soft_update_target_network(self, target, source):
        """Copies weights from source to target with weight [0,1]."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self._hp.target_network_update_factor * param.data +
                                    (1 - self._hp.target_network_update_factor) * target_param.data)

    def _copy_to_target_network(self, target, source):
        """Completely copies weights from source to target."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def _get_optimizer(self, optimizer, model, lr):
        """Returns an instance of the specified optimizers on the parameters of the model with specified learning rate."""
        if optimizer == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optimizer == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optimizer == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optimizer))
        optim = partial(get_optim, gradient_clip=self._hp.gradient_clip)
        return optim(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def _perform_update(self, loss, opt, network):
        """Performs one backward gradient step on the loss using the given optimizer. Also syncs gradients."""
        nan_hook(loss)
        opt.zero_grad()
        loss.backward()

        grads = [p.grad for p in network.parameters()]
        nan_hook(grads)

        opt.step()

    def _get_obs_norm_info(self):
        if isinstance(self._obs_normalizer, DummyNormalizer): return {}
        mean, std = self._obs_normalizer.mean, self._obs_normalizer.std
        if not self._hp.obs_norm_log_groups:
            self._hp.obs_norm_log_groups = AttrDict(all=np.arange(mean.shape[0]))
        info = {}
        for group_key in self._hp.obs_norm_log_groups:
            info['obs_norm_' + group_key + '_mean'] = mean[self._hp.obs_norm_log_groups[group_key]].mean()
            info['obs_norm_' + group_key + '_std'] = std[self._hp.obs_norm_log_groups[group_key]].mean()
        return info

    @staticmethod
    def load_model_weights(model, checkpoint, epoch='latest'):
        """Loads weights for a given model from the given checkpoint directory."""
        checkpoint_dir = checkpoint if os.path.basename(checkpoint) == 'weights' \
                            else os.path.join(checkpoint, 'weights')     # checkpts in 'weights' dir
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model=model)

    @staticmethod
    def _remove_batch(d):
        """Adds batch dimension to all tensors in d."""
        return map_dict(lambda x: x[0] if (isinstance(x, torch.Tensor) or 
                                           isinstance(x, np.ndarray)) else x, d)

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with agent.val_mode(): ...<do something>..."""
        self._is_train = False
        self.call_children("switch_to_val", Policy)
        yield
        self._is_train = True
        self.call_children("switch_to_train", Policy)

    @contextmanager
    def rand_act_mode(self):
        """Performs random actions within context. To be used like: with agent.rand_act_mode(): ...<do something>..."""
        self._rand_act_mode = True
        yield
        self._rand_act_mode = False

    @contextmanager
    def rollout_mode(self):
        """Sets rollout parameters if desired."""
        self._rollout_mode = True
        self.call_children("switch_to_rollout", Policy)
        yield
        self._rollout_mode = False
        self.call_children("switch_to_non_rollout", Policy)

    def call_children(self, fn, cls):
        """Call function with name fn in all submodules of class cls."""
        def conditional_fn(module):
            if isinstance(module, cls):
                getattr(module, fn).__call__()

        self.apply(conditional_fn)


class HierarchicalAgent(BaseAgent):
    """Implements a basic hierarchical agent with high-level and low-level policy/policies."""
    def __init__(self, config):
        super().__init__(config)
        self.hl_agent = self._hp.hl_agent(self._hp.overwrite(self._hp.hl_agent_params))
        self.ll_agent = self._hp.ll_agent(self._hp.overwrite(self._hp.ll_agent_params))
        self._last_hl_output = None     # stores last high-level output to feed to low-level during intermediate steps

    def _default_hparams(self):
        default_dict = ParamDict({
            'hl_agent': None,                         # high-level agent class
            'hl_agent_params': None,                  # parameters of the high-level agent
            'll_agent': None,                         # low-level agent class
            'll_agent_params': None,                  # parameters of the low-level agent(s)
            'update_hl': True,                        # whether to update high-level agent
            'update_ll': True,                        # whether to update low-level agent(s)
            'll_subgoal_reaching_reward': False,      # whether to count ll subgoal reaching reward in training
            'll_subgoal_reaching_reward_weight': 1e3, # weight for the subgoal reaching reward
        })
        return super()._default_hparams().overwrite(default_dict)

    def act(self, obs):
        """Output dict contains is_hl_step in case high-level action was performed during this action."""
        obs_input = obs[None] if len(obs.shape) == 1 else obs    # need batch input for agents
        output = AttrDict()
        if self._perform_hl_step_now:
            # perform step with high-level policy
            self._last_hl_output = self.hl_agent.act(obs_input)
            output.is_hl_step = True
            if len(obs_input.shape) == 2 and len(self._last_hl_output.action.shape) == 1:
                self._last_hl_output.action = self._last_hl_output.action[None]  # add batch dim if necessary
                self._last_hl_output.log_prob = self._last_hl_output.log_prob[None]
        else:
            output.is_hl_step = False
        output.update(prefix_dict(self._last_hl_output, 'hl_'))

        # perform step with low-level policy
        assert self._last_hl_output is not None
        output.update(self.ll_agent.act(self.make_ll_obs(obs_input, self._last_hl_output.action)))

        return self._remove_batch(output) if len(obs.shape) == 1 else output

    def update(self, experience_batches):
        """Updates high-level and low-level agents depending on which parameters are set."""
        assert isinstance(experience_batches, AttrDict)  # update requires batches for both HL and LL
        update_outputs = AttrDict()
        if self._hp.update_hl:
            hl_update_outputs = self.hl_agent.update(experience_batches.hl_batch)
            update_outputs.update(prefix_dict(hl_update_outputs, "hl_"))
        if self._hp.update_ll:
            ll_update_outputs = self.ll_agent.update(experience_batches.ll_batch)
            update_outputs.update(ll_update_outputs)
        return update_outputs

    def log_outputs(self, logging_stats, rollout_storage, logger, log_images, step):
        """Additionally provides option ot visualize hierarchical agents."""
        super().log_outputs(logging_stats, rollout_storage, logger, log_images, step)
        if log_images:
            self.hl_agent.visualize(logger, rollout_storage, step)
            self.ll_agent.visualize(logger, rollout_storage, step)

    def _act_rand(self, obs):
        """Performs random actions with high-level policy. Low-level policy operates normally."""
        with self.hl_agent.rand_act_mode():
            return self.act(obs)

    def make_ll_obs(self, obs, hl_action):
        """Creates low-level agent's observation from env observation and HL action."""
        return np.concatenate((obs, hl_action), axis=-1)

    def add_experience(self, experience_batch):
        self.hl_agent.add_experience(experience_batch.hl_batch)
        self.ll_agent.add_experience(experience_batch.ll_batch)

    def sync_networks(self):
        self.hl_agent.sync_networks()
        self.ll_agent.sync_networks()

    def state_dict(self, *args, **kwargs):
        return {'hl_agent': self.hl_agent.state_dict(*args, **kwargs),
                'll_agent': self.ll_agent.state_dict(*args, **kwargs)}

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.hl_agent.load_state_dict(state_dict.pop('hl_agent'), *args, **kwargs)
        self.ll_agent.load_state_dict(state_dict.pop('ll_agent'), *args, **kwargs)

    def save_state(self, save_dir):
        self.hl_agent.save_state(os.path.join(save_dir, 'hl_agent'))
        self.ll_agent.save_state(os.path.join(save_dir, 'll_agent'))

    def load_state(self, save_dir):
        self.hl_agent.load_state(os.path.join(save_dir, 'hl_agent'))
        self.ll_agent.load_state(os.path.join(save_dir, 'll_agent'))

    def reset(self):
        super().reset()
        self.hl_agent.reset()
        self.ll_agent.reset()

    @contextmanager
    def rand_act_mode(self):
        """Performs random actions within context. To be used like: with agent.rand_act_mode(): ...<do something>..."""
        self._rand_act_mode = True
        self.hl_agent._rand_act_mode = True
        self.ll_agent._rand_act_mode = True
        yield
        self._rand_act_mode = False
        self.hl_agent._rand_act_mode = False
        self.ll_agent._rand_act_mode = False

    @property
    def _perform_hl_step_now(self):
        """Indicates whether the high-level policy should be executed in the current time step."""
        raise NotImplementedError    # should be implemented by child class!


class FixedIntervalHierarchicalAgent(HierarchicalAgent):
    """Hierarchical agent that executes high-level actions in fixed temporal intervals."""
    def __init__(self, config):
        super().__init__(config)
        self._steps_since_hl = 0  # number of steps since last high-level step

    def _default_hparams(self):
        default_dict = ParamDict({
            'hl_interval': 3,       # temporal interval at which high-level actions are executed
        })
        return super()._default_hparams().overwrite(default_dict)

    def act(self, *args, **kwargs):
        output = super().act(*args, **kwargs)
        self._steps_since_hl += 1
        return output

    @property
    def _perform_hl_step_now(self):
        return self._steps_since_hl % self._hp.hl_interval == 0

    def reset(self):
        super().reset()
        self._steps_since_hl = 0     # start new episode with high-level step
