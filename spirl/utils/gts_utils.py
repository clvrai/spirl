
import numpy as np
from gym_gts import GTSApi
import gym

#  =========================== env setup utils ================================
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

# tire type: 
# CH, CM, CS, (comfort)
# SH, SM, SS, (sport)
# RH, RM, RS, RSS, (racing hard)
# IM, HW, or DIRT (wet tire)

TIRE_TYPE = 'SH'

RL_OBS_1 = ['lap_count', 'current_lap_time_msec', 'speed_kmph', 'frame_count', 'is_controllable',
                'vx', 'vy', 'vz', 'pos','rot', 'angular_velocity', 'front_g', 'side_g', 'vertical_g',
                'centerline_diff_angle', 'centerline_distance', 'edge_l_distance', 'edge_r_distance', 'course_v',
                'is_hit_wall', 'is_hit_cars', 'hit_wall_time', 'hit_cars_time',
                'steering', 'throttle', 'brake'] + \
                ["curvature_in_%.1f_%s" % (step, "seconds") for step in np.arange(start=1.0, stop=3.0, step=0.2)] \
                + ["lidar_distance_%i_deg" % deg for deg in np.concatenate(
                (np.arange(start=0, stop=105, step=15), np.arange(start=270, stop=360, step=15),))]

def start_condition_formulator(num_cars, course_v, speed):
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


def BoP_formulator(num_cars, car_name, weight_percentage, power_percentage):
    bop_base = BOP[car_name]
    power_base = bop_base['power']
    weight_base = bop_base['weight']

    bops = []
    for car_id in range(num_cars):
        bops.append( {"enable": True, 
                        "power": round(power_base * power_percentage[car_id]), 
                        "weight": round(weight_base * weight_percentage[car_id]) } )

    return bops

def initialize_gts(ip, num_cars, car_codes, course_code, tire_type, bops):
    with GTSApi(ip=ip) as gts_api:
        gts_api.set_race(
            num_cars = num_cars,
            car_codes = car_codes,
            course_code = course_code,
            front_tires = tire_type,
            rear_tires = tire_type,
            bops = bops
        )

def make_env(ip, min_frames_per_action, feature_keys, builtin_controlled, spectator_mode=False):
    env = gym.make(
                'gts-v0', 
                ip=ip,  
                min_frames_per_action=min_frames_per_action,
                builtin_controlled = builtin_controlled,
                feature_keys = feature_keys,
                standardize_observations = False,
                store_states = False,
                spectator_mode = spectator_mode
        )
    return env



#  =================================== state observe ===============================
def WrapToPi(x):
    x = np.mod(x, 2*np.pi)
    x -= 2*np.pi * np.heaviside((x - np.pi), 0)
    return x

def gts_state_2_cartesian(gts_state):
    state = {}
    # to store rotation at first
    if 'rot' in gts_state:
        state.update(gts_state_2_cartesian( [{'rot':gts_state['rot']}] ) )

    for key in gts_state:
        # car gts frame states 
        if key == 'current_lap_time_msec':
            state['t'] = gts_state[key]/1000
        elif key == 'speed_kmph':
            state['V'] = gts_state[key] / 3.6
        elif key == 'steering':
            state['delta'] = gts_state[key]
        elif key == 'throttle':
            state['thr'] = gts_state[key]
        elif key == 'brake':
            state['brk'] = gts_state[key]

        # car states in frenet coordinate
        elif key == 'centerline_diff_angle':
            state['epsi'] = gts_state[key]
        elif key == 'centerline_distance':
            state['ey'] = - gts_state[key]
        elif key == 'course_v':
            state['s'] = gts_state['course_v']
        elif key == 'frame_count':
            state['frame_count'] = gts_state[key]
        elif key == 'lap_count':
            state['lap_count'] = gts_state['lap_count']

        elif key == 'edge_l_distance':
            state['Wl'] = gts_state['edge_l_distance']
        elif key == 'edge_r_distance':
            state['Wr'] = gts_state['edge_r_distance']

        # car chassis state in cartesian coordinate
        elif key == 'pos':
            state['X'] = gts_state['pos'][0]
            state['Y'] = gts_state['pos'][2]
            state['Z'] = gts_state['pos'][1]
        elif key == 'rot': # pitch yaw roll
            state['Theta'] = - gts_state['rot'][0]
            state['Psi'] = WrapToPi( np.pi/2 - gts_state['rot'][1] )
            state['Phi'] = gts_state['rot'][2]
        elif key == 'vx':
            state['Vx'] = gts_state['vx']
        elif key == 'vy':
            state['Vy'] = gts_state['vy']
        elif key == 'vz':
            state['Vz'] = gts_state['vz']

        # elif key == 'velocity' and 'vx' not in gts_state and 'vy' not in gts_state:
        #     if 'Psi' in state:
        #         Psi = state['Psi']
        #     else:
        #         Psi = WrapToPi( np.pi/2 - gts_state['rot'][1] )
        #     # state['velocity'] = gts_state['velocity']
        #     v1 = velocity[0]
        #     v2 = velocity[2]
        #     state['Vx'] = v1*np.cos(Psi) + v2*np.sin(Psi)
        #     state['Vy'] = -v1*np.sin(Psi) + v2*np.cos(Psi)

        elif key == 'angular_velocity':
            state['dpsi'] = - gts_state['angular_velocity'][1] # roll, pitch, yaw
            state['dtheta'] = - gts_state['angular_velocity'][0]
            state['dphi'] = gts_state['angular_velocity'][2]
        elif key == 'front_g':
            state['ax'] = gts_state['front_g']
        elif key == 'side_g':
            state['ay'] = - gts_state['side_g']
        elif key == 'vertical_g':
            state['az'] = gts_state['vertical_g']

        # car tire states
        # 0 front left, 1 front right, 2 rear left, 3 rear right
        elif key == 'wheel_load':
            state['Fz4'] = gts_state['wheel_load']
            state['Fzf'] = sum(gts_state['wheel_load'][0:2])
            state['Fzr'] = sum(gts_state['wheel_load'][2:4])
            state['Fz'] = sum(gts_state['wheel_load'])
        elif key == 'slip_angle':
            state['alpha4'] = gts_state['slip_angle']
            state['alphaf'] = np.mean(gts_state['slip_angle'][0:2])
            state['alphar'] = np.mean(gts_state['slip_angle'][2:4])
        elif key == 'slip_ratio':
            state['sigma'] = gts_state['slip_ratio']
            state['sigmaf'] = np.mean(gts_state['slip_ratio'][0:2])
            state['sigmar'] = np.mean(gts_state['slip_ratio'][2:4])
        elif key == 'wheel_angle':
            state['delta4'] = gts_state['wheel_angle']
            state['deltaf'] = np.mean(gts_state['wheel_angle'][0:2])
            state['deltar'] = np.mean(gts_state['wheel_angle'][2:4])
        elif key == 'wheel_omega':
            state['omega4'] = gts_state['wheel_omega']
            state['omegaf'] = np.mean(gts_state['wheel_omega'][0:2])
            state['omegar'] = np.mean(gts_state['wheel_omega'][2:4])

        # engine states
        elif key == 'engine_torque':
            state['Te'] = gts_state['engine_torque']

        # hit
        elif key == 'is_hit_wall':
            state['hit_wall'] = 1 if gts_state['is_hit_wall'] else 0
        elif key == 'is_hit_cars':
            state['hit_car'] = 1 if gts_state['is_hit_cars'] else 0
        elif key == 'hit_wall_time':
            state['hit_wall_time'] = gts_state['hit_wall_time']
        elif key == 'hit_cars_time':
            state['hit_cars_time'] = gts_state['hit_cars_time']
        elif key == 'shift_position':
            state['shift'] = gts_state['shift_position']
        elif key == 'is_controllable':
            state['controllable'] = 1 if gts_state['is_controllable'] else 0

        elif 'curvature' in key:
            state[key] = gts_state[key]
        elif 'lidar' in key:
            state[key] = gts_state[key]

        else:
            pass

    return state