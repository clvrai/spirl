from ast import Gt
from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import RL_OBS_1, CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP

class GTSEnv(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        ip_address = '192.168.124.14'
        car_name = 'Audi TTCup'
        course_name = 'Tokyo Central Outer' 

        spectator_mode = False

        bops = [BOP[car_name]] * 2
        initialize_gts(ip = ip_address,
                      num_cars=2, 
                      car_codes = CAR_CODE[car_name], 
                      course_code = COURSE_CODE[course_name], 
                      tire_type = TIRE_TYPE, 
                      bops = bops
                      )

        self._env = make_env(
            ip = ip_address, 
            min_frames_per_action=6, 
            feature_keys = RL_OBS_1, 
            builtin_controlled = [1], 
            spectator_mode=spectator_mode
        )

        self.course_length = self._get_course_length()
    
    def _default_hparams(self):
        default_dict = ParamDict({})
        return super()._default_hparams().overwrite(default_dict)

    def reset(self, start_conditions=None):
        obs = self._env.reset(start_conditions=start_conditions)
        return self._wrap_observation(obs)

    def step(self, actions):
        obs, rew, done, info = self.env_.step(actions)
        return self._wrap_observation(obs), rew, done, info

    def _wrap_observation(self, obs):
        return obs

    def _get_course_length(self):
        course_length, course_code, course_name = self.env.get_course_meta()
        return course_length

if __name__ == "__main__":
    from spirl.utils.general_utils import AttrDict
    conf = AttrDict()
    env  = GTSEnv(conf)
    env.reset()