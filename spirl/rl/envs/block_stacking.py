import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.data.block_stacking.src.block_stacking_env import BlockStackEnv as UnwrappedBlockStackEnv
from spirl.data.block_stacking.src.block_stacking_env import HighStackBlockStackEnv, SparseHighStackBlockStackEnv
from spirl.data.block_stacking.src.block_task_generator import FixedSizeSingleTowerBlockTaskGenerator


class BlockStackEnv(GymEnv):
    """Tiny wrapper around GymEnv for Block Stacking tasks."""

    def _default_hparams(self):
        default_dict = ParamDict({
            'env_config': None,     # override env_config if desired
        })

        return super()._default_hparams().overwrite(default_dict)

    def _get_default_env_config(self):
        default_task_params = AttrDict(
            max_tower_height=4
        )

        default_env_config = AttrDict(
            task_generator=FixedSizeSingleTowerBlockTaskGenerator,
            task_params=default_task_params,
            dimension=2
        )
        return default_env_config

    def _make_env(self, name):
        default_env_config = self._get_default_env_config()
        if self._hp.env_config is not None:
            default_env_config.update(self._hp.env_config)

        return UnwrappedBlockStackEnv(default_env_config)

    @property
    def agent_params(self):
        return self._env.agent_params


class Stack4BlockStackEnvV0(BlockStackEnv):
    DEFAULT_QUAT = np.array([0.70710678, 0, 0, -0.70710678])
    TASK = [(2, 3), (3, 1), (1, 4), (4, 0)]
    BLOCK_POS = [
        AttrDict(pos=np.array([0, -0.4]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, -0.2]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.0]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.2]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.4]), quat=DEFAULT_QUAT)
    ]

    def _get_default_env_config(self):
        default_env_config = super()._get_default_env_config()
        default_env_config.fixed_task = self.TASK
        default_env_config.fixed_block_pos = self.BLOCK_POS
        return default_env_config


class HighStackStackEnvV0(Stack4BlockStackEnvV0):
    def _make_env(self, name):
        default_env_config = self._get_default_env_config()
        if self._hp.env_config is not None:
            default_env_config.update(self._hp.env_config)

        return HighStackBlockStackEnv(default_env_config)


class HighStack11StackEnvV0(HighStackStackEnvV0):
    DEFAULT_QUAT = np.array([0.70710678, 0, 0, -0.70710678])
    TASK = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    BLOCK_POS = [
        AttrDict(pos=np.array([0, -1.0]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, -0.8]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, -0.6]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, -0.4]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, -0.2]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.0]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.2]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.4]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.6]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 0.8]), quat=DEFAULT_QUAT),
        AttrDict(pos=np.array([0, 1.0]), quat=DEFAULT_QUAT),
    ]

    def _get_default_env_config(self):
        default_env_config = super()._get_default_env_config()
        default_env_config.fixed_task = self.TASK
        default_env_config.fixed_block_pos = self.BLOCK_POS
        default_env_config.table_size = (1.2, 2.4, 0.8)
        default_env_config.n_blocks = 11
        return default_env_config


class SparseHighStack11StackEnvV0(HighStack11StackEnvV0):
    def _make_env(self, name):
        default_env_config = self._get_default_env_config()
        if self._hp.env_config is not None:
            default_env_config.update(self._hp.env_config)

        return SparseHighStackBlockStackEnv(default_env_config)
