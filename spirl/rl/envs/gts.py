from ast import Gt
from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import RL_OBS_1, CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP
from spirl.utils.gts_utils import raw_observation_to_true_observation

class GTSEnv_Base(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._hp.overwrite(self._game_hp())

        if (self._hp.do_init):
            self._initialize()
        self._make_env()

    
    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.124.14',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 1,
            'spectator_mode' : False
        })
        return super()._default_hparams().overwrite(default_dict)

    def _game_hp(self):
        game_hp = ParamDict({
            'builtin_controlled' : [],
            'do_init' : False
        })
        return game_hp

    def _initialize(self):
        bops = [BOP[self._hp.car_name]] * self._hp.num_cars

        initialize_gts(ip = self._hp.ip_address,
                      num_cars=self._hp.num_cars, 
                      car_codes = CAR_CODE[self._hp.car_name], 
                      course_code = COURSE_CODE[self._hp.course_name], 
                      tire_type = TIRE_TYPE, 
                      bops = bops
                      )
    
    def _make_env(self):
        self._env = make_env(
            ip = self._hp.ip_address, 
            min_frames_per_action=6, 
            feature_keys = RL_OBS_1, 
            builtin_controlled = self._hp.builtin_controlled, 
            spectator_mode = self._hp.spectator_mode
        )

        self.course_length = self._get_course_length()

    def reset(self, start_conditions=None):

        obs = self._env.reset(start_conditions=start_conditions)
        return self._wrap_observation(obs)

    def step(self, actions):
        obs, rew, done, info = self._env.step(actions)
        return self._wrap_observation(obs), rew, done, info

    def _wrap_observation(self, obs):
        return raw_observation_to_true_observation(obs[0])

    def _get_course_length(self):
        course_length, course_code, course_name = self._env.get_course_meta()
        return course_length

if __name__ == "__main__":
    from spirl.utils.general_utils import AttrDict
    conf = AttrDict()
    env  = GTSEnv_Base(conf)
    obs = env.reset()
    obs, rew, done, info = env.step([[0, -1]])
