from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.rl.envs.gts import GTSEnv_Base
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import RL_OBS_1, CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP
from spirl.utils.gts_utils import raw_observation_to_true_observation

from spirl.utils.gts_utils import reward_function, sampling_done_function

class GTSEnv_Multi(GTSEnv_Base):


    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.124.14',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 20,
            'spectator_mode' : False,
        })
        return super()._default_hparams().overwrite(default_dict)

    

    