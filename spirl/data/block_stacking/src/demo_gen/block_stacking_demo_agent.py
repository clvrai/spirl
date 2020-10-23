from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.rl.components.agent import BaseAgent

from spirl.data.block_stacking.src.demo_gen.block_demo_policy import ClosedLoopBlockStackDemoPolicy


class BlockStackingDemoAgent(BaseAgent):
    """Wraps demo policy for block stacking."""
    def __init__(self, config):
        super().__init__(config)
        self._policy = self._hp.policy(self._hp.env_params)

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy': ClosedLoopBlockStackDemoPolicy,         # policy class
            'env_params': None,                     # parameters containing info about env -> set automatically
        })
        return super()._default_hparams().overwrite(default_dict)

    @property
    def rollout_valid(self):
        return self._hp.env_params.task_complete_check()

    def reset(self):
        self._policy.reset()

    def _act(self, obs):
        return AttrDict(action=self._policy.act(obs))

    def _act_rand(self, obs):
        raise NotImplementedError("This should not be called in the demo agent.")
