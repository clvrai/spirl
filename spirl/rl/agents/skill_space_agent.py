from collections import deque
import contextlib

import numpy as np

from spirl.rl.components.agent import BaseAgent
from spirl.utils.general_utils import ParamDict, split_along_axis, AttrDict
from spirl.utils.pytorch_utils import map2torch, map2np, no_batchnorm_update


class SkillSpaceAgent(BaseAgent):
    """Agent that acts based on pre-trained VAE skill decoder."""
    def __init__(self, config):
        super().__init__(config)
        self._update_model_params()     # transfer some parameters to model

        self._policy = self._hp.model(self._hp.model_params, logger=None)
        self.load_model_weights(self._policy, self._hp.model_checkpoint, self._hp.model_epoch)

        self.action_plan = deque()

    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,              # policy class
            'model_params': None,       # parameters for the policy class
            'model_checkpoint': None,   # checkpoint path of the model
            'model_epoch': 'latest',    # epoch that checkpoint should be loaded for (defaults to latest)
        })
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        assert len(obs.shape) == 2 and obs.shape[0] == 1  # assume single-observation batches with leading 1-dim
        if not self.action_plan:
            # generate action plan if the current one is empty
            split_obs = self._split_obs(obs)
            with no_batchnorm_update(self._policy) if obs.shape[0] == 1 else contextlib.suppress():
                actions = self._policy.decode(map2torch(split_obs.z, self._hp.device),
                                              map2torch(split_obs.cond_input, self._hp.device),
                                              self._policy.n_rollout_steps)
            self.action_plan = deque(split_along_axis(map2np(actions), axis=1))
        return AttrDict(action=self.action_plan.popleft())

    def reset(self):
        self.action_plan = deque()      # reset action plan

    def update(self, experience_batch):
        return {}    # TODO(karl) implement finetuning for policy

    def _split_obs(self, obs):
        assert obs.shape[1] == self._policy.state_dim + self._policy.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self._policy.latent_dim],   # condition decoding on state
            z=obs[:, -self._policy.latent_dim:],
        )

    def sync_networks(self):
        pass        # TODO(karl) only need to implement if we implement finetuning

    def _update_model_params(self):
        self._hp.model_params.device = self._hp.device  # transfer device to low-level model
        self._hp.model_params.batch_size = 1            # run only single-element batches

    def _act_rand(self, obs):
        return self._act(obs)


class ACSkillSpaceAgent(SkillSpaceAgent):
    """Unflattens prior input part of observation."""
    def _split_obs(self, obs):
        unflattened_obs = map2np(self._policy.unflatten_obs(
            map2torch(obs[:, :-self._policy.latent_dim], device=self.device)))
        return AttrDict(
            cond_input=unflattened_obs.prior_obs,
            z=obs[:, -self._policy.latent_dim:],
        )
