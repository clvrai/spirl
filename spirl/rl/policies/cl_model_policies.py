import torch
import numpy as np

from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.utils.pytorch_utils import no_batchnorm_update, ar2ten, ten2ar
from spirl.rl.components.agent import BaseAgent
from spirl.rl.components.policy import Policy
from spirl.modules.variational_inference import MultivariateGaussian


class ClModelPolicy(Policy):
    """Initializes policy network with pretrained closed-loop skill decoder."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.update_model_params(self._hp.policy_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_model': None,              # policy model class
            'policy_model_params': None,       # parameters for the policy model
            'policy_model_checkpoint': None,   # checkpoint path of the policy model
            'policy_model_epoch': 'latest',    # epoch that checkpoint should be loaded for (defaults to latest)
            'load_weights': True,              # optionally allows to *not* load the weights (ie train from scratch)
            'initial_log_sigma': -50,          # initial log sigma of policy dist (since model is deterministic)
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        with no_batchnorm_update(self):     # BN updates harm the initialized policy
            return super().forward(obs)

    def _build_network(self):
        net = self._hp.policy_model(self._hp.policy_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.policy_model_checkpoint, self._hp.policy_model_epoch)
        self._log_sigma = torch.tensor(self._hp.initial_log_sigma * np.ones(self.action_dim, dtype=np.float32),
                                       device=self.device, requires_grad=True)
        return net

    def _compute_action_dist(self, obs):
        assert len(obs.shape) == 2
        split_obs = self._split_obs(obs)
        if obs.shape[0] == 1:
            # during rollouts use HL z every H steps and execute LL policy every step
            if self.steps_since_hl > self.horizon - 1:
                self.last_z = split_obs.z
                self.steps_since_hl = 0
            act = self.net.decoder(torch.cat((split_obs.cond_input, self.last_z), dim=-1))
            self.steps_since_hl += 1
        else:
            # during update (ie with batch size > 1) recompute LL action from z
            act = self.net.decoder(torch.cat((split_obs.cond_input, split_obs.z), dim=-1))
        return MultivariateGaussian(mu=act, log_sigma=self._log_sigma[None].repeat(act.shape[0], 1))

    def sample_rand(self, obs):
        if len(obs.shape) == 1:
            output_dict = self.forward(obs[None])
            output_dict.action = output_dict.action[0]
            return output_dict
        return self.forward(obs)    # for prior-initialized policy we run policy directly for rand sampling from prior

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self.net.latent_dim],   # condition decoding on state
            z=obs[:, -self.net.latent_dim:],
        )

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        params.batch_size = 1  # run only single-element batches for forward pass

    @property
    def horizon(self):
        return self._hp.policy_model_params.n_rollout_steps


class ACClModelPolicy(ClModelPolicy):
    """Handles image observations in ClModelPolicy."""
    def _split_obs(self, obs):
        unflattened_obs = self.net.unflatten_obs(obs[:, :-self.net.latent_dim])
        return AttrDict(
            cond_input=self.net.enc_obs(unflattened_obs.prior_obs),
            z=obs[:, -self.net.latent_dim:],
        )
