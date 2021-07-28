import numpy as np
from torchvision.transforms import Resize
from PIL import Image

import roboverse
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.rl.components.environment import GymEnv


class OfficeEnv(GymEnv):
    """Tiny wrapper around gym env for WidowX roboverse env."""
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "Widow250OfficeFixed-v0",
        }))

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return obs, np.float64(rew), done, info

    def _wrap_observation(self, obs):
        return np.asarray(obs['state'], dtype=np.float32)  # [52:62]

    def _render_raw(self, mode):
        """Returns rendering as uint8 in range [0...255]"""
        assert mode == 'rgb_array'  # currently only rgb array is supported
        return self._env.render_obs(res=self._hp.screen_height).transpose(1, 2, 0)   # HACK, make this proper res


class OfficeImageEnv(OfficeEnv):
    """Observation is rendered, flattened image."""
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'obs_res': 64,      # resolution of image observation
        }))

    def _wrap_observation(self, obs):
        img = Resize((self._hp.obs_res, self._hp.obs_res))\
                (Image.fromarray(np.asarray(obs['image'].reshape(48,48,3) * 255., dtype=np.uint8)))
        return (np.asarray(img, dtype=np.float32).transpose(2,0,1) / 255. * 2 - 1).flatten()

