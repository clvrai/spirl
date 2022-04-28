from ast import Gt
from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

CAR_CODE = {'Mazda Roadster':   2148, 
            'Mazda Demio':      3383, 
            'Audi TTCup':       3298
            }

COURSE_CODE = { 'Tokyo Central Outer':      351, 
                'Tokyo East Outer':         361, 
                'Tokyo Central Inner':      450, 
                'Brandas Hatch':            119,
                'protect':                  452
            }

BOP     =   {
            'default':      {"enable": False, "power": 100, "weight": 100},
            'Mazda Demio':  {"enable": True, "power": 124, "weight": 119},
            'Audi TTCup' :  {"enable": True, "power": 104, "weight": 97},
            }

RL_OBS_1 = ['lap_count', 'current_lap_time_msec', 'speed_kmph', 'frame_count', 'is_controllable',
                'vx', 'vy', 'vz', 'pos','rot', 'angular_velocity', 'front_g', 'side_g', 'vertical_g',
                'centerline_diff_angle', 'centerline_distance', 'edge_l_distance', 'edge_r_distance', 'course_v',
                'is_hit_wall', 'is_hit_cars', 'hit_wall_time', 'hit_cars_time',
                'steering', 'throttle', 'brake'] + \
                ["curvature_in_%.1f_%s" % (step, "seconds") for step in np.arange(start=1.0, stop=3.0, step=0.2)] \
                + ["lidar_distance_%i_deg" % deg for deg in np.concatenate(
                (np.arange(start=0, stop=105, step=15), np.arange(start=270, stop=360, step=15),))]


class GTSEnv(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env()
    
    def _default_hparams(self):
        default_dict = ParamDict({})
        return super()._default_hparams().overwrite(default_dict)

    def reset(self, start_conditions):

        obs = self._env.reset(start_conditions)
        return self._wrap_observation(obs)

    def initialize_setting(self):

        bop = BOP[self._hp['car_name']]

        with GTSApi(ip=self._hp['ip_address']) as gts_api:
            gts_api.set_race(
                num_cars = self._hp['num_cars'],
                car_codes = CAR_CODE[self._hp['car_name']],
                course_code = COURSE_CODE[self._hp['course_name']],
                front_tires = self._hp['tire_type'],
                rear_tires = self._hp['tire_type'],
                bops = [bop]
            )

    def step(self, actions):
        obs, rew, done, info = self.env_.step(actions)
        return self._wrap_observation(obs), rew, done, info

    def _wrap_observation(self, obs):
        return obs

    def start_condition_formulator(self, num_cars, course_v, speed):
        conditions = []
        for car_id in range(num_cars):
            conditions.append(
                {
                    "id": car_id,
                    "course_v": course_v[car_id],
                    "speed_kmph": speed[car_id],
                    "angular_velocity": 0,
                }
            )

        start_conditions = {"launch": conditions}
        return start_conditions

    def _make_env(self):

        builtin_controlled = [0, 1]
        env = gym.make(
                'gts-v0', 
                ip=self._hp['ip_address'],  
                min_frames_per_action=6,

                builtin_controlled = builtin_controlled,

                feature_keys = RL_OBS_1,
                standardize_observations = False,
                store_states = False,
                spectator_mode = False
        )
        return env

    def _get_course_length(self):
        course_length, course_code, course_name = self.env.get_course_meta()
        return course_length

if __name__ == "__main__":
    ip_address = '192.168.124.14'
    # from spirl.configs.rl.gts.base_conf import configuration
    # env = GTSEnv(configuration)
    # env._get_course_length()

    # with GTSApi(ip=ip_address) as gts_api:
    #     gts_api.set_race(
    #                 num_cars = 1,
    #                 car_codes = 2148,
    #                 course_code = 351
    #             )

    env = gym.make(
                    'gts-v0', 
                    ip=ip_address,  
                    min_frames_per_action=1
                )
    course_length, course_code, course_name = env.get_course_meta()
    print('course length 2',course_length)
