import numpy as np
import copy

from spirl.data.block_stacking.src.robosuite.models import MujocoWorldBase
from spirl.data.block_stacking.src.robosuite.models.arenas import TableArena
from spirl.data.block_stacking.src.robosuite.utils.mjcf_utils import new_joint, new_geom, new_body

from spirl.rl.components.environment import BaseEnvironment
from spirl.rl.utils.robosuite_utils import FastResetMujocoEnv
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.data.block_stacking.src.utils.utils import lookat_to_quat, convert_quat
from spirl.data.block_stacking.src.utils.block import Block
from spirl.data.block_stacking.src.utils.block_placement_helper import PlacementGenerator
from spirl.data.block_stacking.src.utils.gripper_only_robot import GripperOnlyRobot
from spirl.data.block_stacking.src.utils.wide_range_gripper import WideRangeGripper
from spirl.data.block_stacking.src.utils.numbered_box_object import NumberedBoxObject

color_dict = {
    "blue": (np.array([66,133,244,255]) / 255).tolist(),
    "red": (np.array([219,68,55,255]) / 255).tolist(),
    "yellow": (np.array([244,160,0,255]) / 255).tolist(),
    "green": (np.array([15,157,88,255]) / 255).tolist(),
    "white": [1., 1., 1., 1.]
}


class BlockStackEnv(FastResetMujocoEnv, BaseEnvironment):
    """2D blockstacking env."""
    DENSE_REWARD_SCALING_FACTOR = 2       # controls wideness of attraction basin (low factor = wide)
    RELATIVE_REWARD_SCALE = 1 / 0.05      # factor for relative shaped reward computation
    STACKED_REWARD = 2.0                  # reward value for stacking one block on another block
    DELIVERED_REWARD = 0.75               # reward for delivering the block above the target block
    LIFTED_REWARD = 0.25                  # reward for lifting a block above threshold (aka grasped)
    REACHED_REWARD = 0.125                # reward for reaching a block

    def __init__(self, hp):
        self._hp = self._default_hparams().overwrite(hp)
        self._blocks = []   # list of block objects

        self.table_full_size = self._hp.table_size
        self.robot_pos = [0, 0, 0]
        self.arena_pos = [0, 0, 0]

        self._task_gen = self._hp.task_generator(self._hp.task_params, self._hp.n_blocks)
        self._fixed_task = self._task_gen.sample()   # this is used if task stays fixed
        self._task = None
        self._last_gripper_action = None        # for computing action penalty for absolute action
        if self._hp.fixed_task is not None:
            assert not self._hp.rand_task
            self._fixed_task = self._hp.fixed_task

        self._placement_gen = PlacementGenerator(self._hp.task_params.seed if "seed" in self._hp.task_params else
                                                 self._hp.seed)
        self._init_block_placements = self._hp.fixed_block_pos if self._hp.fixed_block_pos is not None else None
        self._block_placements = None

        self._init_qpos = None

        # internally saved episode goal to avoid regeneration within episode
        self._goal = None

        # internal state to help reward calculation
        self._reached_flag = None
        self._lifted_flag = None
        self._delivered_flag = None
        self._stack_flag = None
        self._ep_dict = AttrDict({})

        self._t = 0

        super().__init__()

        self._prev_block_pos = [copy.deepcopy(b.pos) for b in self._blocks]  # list of prev block objects (for relative reward computation)
        self._prev_gripper_pos = copy.deepcopy(self.gripper_pos)
        #self.sim.model.opt.timestep = 0.002 * 2  # this could be used to improve efficiency by increasing sim step size

    def _default_hparams(self):
        return BaseEnvironment._default_hparams(self).overwrite(ParamDict({
            'n_blocks': 5,                 # number of blocks in env
            'block_size': 0.04,            # size of a block
            'block_color': 'white',        # color of the block
            'rotate_blocks': False,        # no block rotation if set to False
            'allow_rotate': False,         # if False, disallow gripper rotation
            'table_size': (1.2, 1.2, 0.8), # size of table
            'dimension': 2,                # dimensionality for the task
            'camera_name': 'frontview',    # name of camera to render
            'gripper_width': 0.02,         # thickness of gripper to consider during placement
            'task_generator': None,        # task generator for generating HL plans
            'task_params': AttrDict({}),   # parameters for task generator
            'perturb_actions': False,      # if True, perturb action and init block placement
            'perturb_prob': 0.3,           # action perturb probability
            'perturb_scale': 0.03,         # action perturb scale
            'n_steps': None,               # number of steps in the task, default n_blocks - 1
            'friction': 1,                 # friction for the boxes
            'rand_task': False,            # if True, randomizes the task in every reset (i.e. multi-task env)
            'rand_init_pos': False,        # if False, keeps initial position of blocks constant
            'rand_init_gripper': False,    # if True, randomizes gripper xy position at each episode reset
            'include_quat': False,         # if True, include quaternions in observation
            'include_vel': True,           # if True, include velocity of the gripper
            'include_2d_rotation': False,  # if True, adds 2D rotation representation for blocks to obs (sin+cos)
            'clip_obs': 2.0,               # if not None, clip observation values
            'seed': None,                  # seed for generating block placements
            'relative_shaped_reward': False,  # if True, computes shaping reward as relative change towards the goal
            'action_penalty_weight': 0.,   # penalty for action magnitude
            'reward_density': 'dense',     # integer defining how dense the reward is ['dense', 'sparse']
            'number_blocks': False,        # if True, print number on blocks
            'fixed_task': None,            # (optional) if provided is used as fixed task
            'fixed_block_pos': None,       # (optional) if provided is used as fixed block position
            'add_boundary_walls': True,    # if True, adds invisible walls that constrain movement
            'reset_with_boundary': False,  # if True, resets episode once agent leaves allowed region

            'reward_scale': 1.0,           # scale of the reward
        }))

    def reset(self):
        if self._init_qpos is None:
            obs = super().reset()
            self._init_qpos = self.sim.data.qpos.copy()
            return obs
        else:
            self.sim.data.qpos[:len(self._init_qpos)] = self._init_qpos.copy()
            self._reset_internal()
            self.sim.forward()
            return self._get_observation()

    # @timed("Step ")
    def step(self, action):
        """Step the environment with symmetric gripper movements."""
        # process action
        raw_action = action
        action = self._pad_action(action)
        real_action = np.zeros((len(action) + 1,))
        real_action[:len(action)] = action
        real_action[-1] = self._adjust_gripper_finger_action(action[-1])
        real_action[-2] = -real_action[-1]
        if self._hp.perturb_actions and np.random.rand() < self._hp.perturb_prob:
            real_action += np.random.normal(0, self._hp.perturb_scale, real_action.shape[0])

        # step through environment
        # with timing("Step raw "):
        obs, rew, done, info = super().step(real_action)

        # apply action penalty
        if self._hp.action_penalty_weight > 0.0:
            action_penalty = self._hp.action_penalty_weight * self._compute_action_penalty(action)
            rew -= action_penalty
            info.update(AttrDict(action_penalty=action_penalty))

        # episode done
        if self.task_complete():
            done = True

        # terminate episode if gripper or object not on arena
        if self._hp.reset_with_boundary:
            unflattened_obs = self._unflatten_block_obs(obs)
            gripper_pos = unflattened_obs.gripper_pos
            if self._position_invalid(gripper_pos):
                done = True
            for block in self._blocks:
                if self._position_invalid(block.pos):
                    done = True

        # internal variable updates
        self._t += 1
        self._last_gripper_action = action[4]

        # add original action
        info.update(AttrDict(
            raw_action=np.array(raw_action).round(3)
        ))

        return obs, rew, done, info

    def render(self, mode="rgb_array", camera_name=None):
        # set camera position
        agentview_camera_id = self.sim.model.camera_name2id('agentview')
        self._set_camera_position(agentview_camera_id, self.gripper_pos + np.array([0.6, 0, -0.1]))
        self._set_camera_rotation(agentview_camera_id, self.gripper_pos + np.array([0, 0, -0.2]))
        self.sim.forward()

        # render image
        raw_img = self.sim.render(camera_name=camera_name if camera_name is not None else self._hp.camera_name,
                                  height=self._hp.screen_height,
                                  width=self._hp.screen_width,
                                  depth=False)
        # known bug that sim renderer returns flipped images  --  https://github.com/StanfordVL/robosuite/issues/56
        return np.flip(raw_img, axis=0)

    def _reset_internal(self, keep_sim_object=False):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset goal
        self._goal = None

        # reset episode time
        self._t = 0

        # reward at previous timestep
        self._prev_reward = 0.0

        # reset remembered prev finger action
        self._prev_gripper_finger_action = None

        # reset task
        # if self._task is None or self._hp.rand_task:
        self._task = self._task_gen.sample() if self._hp.rand_task else copy.deepcopy(self._fixed_task)

        # reset position of gripper
        self._place_gripper()

        # reset positions of objects
        [b.reset() for b in self._blocks]
        self._place_objects()

        # reset robot joint positions
        self.sim.data.qpos[self.robot_joint_indexes] = np.array(self.mujoco_robot.init_qpos)

        # reset gripper joint positions
        self.sim.data.qpos[self.gripper_joint_indexes] = np.array(self.gripper.init_qpos)

        # reset reward internal states
        self._reached_flag = [False] * len(self._task)
        self._lifted_flag = [False] * len(self._task)
        self._delivered_flag = [False] * len(self._task)
        self._placed_flag = [False] * len(self._task)
        self._stacked_flag = [False] * len(self._task)

    def _place_gripper(self):
        gripper_position = np.array([0, 0, 1.24])
        if self._hp.rand_init_gripper:
            if self._hp.dimension == 2:
                gripper_y = np.random.rand() * self._hp.table_size[1] - self._hp.table_size[1] / 2
                gripper_position[1] = gripper_y
            else:
                gripper_x = np.random.rand() * self._hp.table_size[0] - self._hp.table_size[0] / 2
                gripper_y = np.random.rand() * self._hp.table_size[1] - self._hp.table_size[1] / 2
                gripper_position[:2] = [gripper_x, gripper_y]
        self.sim.data.qpos[self.robot_joint_indexes[0]:self.robot_joint_indexes[0] + 3] = gripper_position

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        """
        di = super()._get_observation()

        # low-level object information
        block_pos = np.array([block.pos for block in self._blocks])
        block_quat = np.array([block.quat for block in self._blocks])
        di["block_pos"] = block_pos
        di["block_quat"] = block_quat

        # robot pose
        di["gripper_pos"] = self.gripper_pos
        di["gripper_vel"] = self.gripper_vel
        di["gripper_angle"] = self.gripper_angle
        di["gripper_finger_pos"] = self.gripper_finger_pos

        ret = self._flatten_block_obs(di)

        # hack to add block orientation
        if self._hp.include_2d_rotation:
            ret = np.concatenate([ret] + [[np.sin(b.z_angle), np.cos(b.z_angle)] for b in self._blocks])

        if self._hp.clip_obs is not None and self._hp.clip_obs > 0:
            ret = np.clip(ret, -self._hp.clip_obs, self._hp.clip_obs)

        return ret

    def _get_reward(self):
        """
        Based on input task definition this can compute a semi-dense subgoal-based reward.
        """
        rew_dict = AttrDict()
        total_rew = 0.0

        prev_block_stacked = True
        # print("{} - {}".format(self.gripper_pos, self.gripper_finger_pos))

        for i, (bottom_block_idx, top_block_idx) in enumerate(self._task):
            bottom_block, top_block = self._blocks[bottom_block_idx], self._blocks[top_block_idx]
            prev_bottom_block_pos, prev_top_block_pos = self._prev_block_pos[bottom_block_idx], self._prev_block_pos[top_block_idx]

            rew = 0.0

            reach_reward, lift_reward, deliver_reward, stack_reward = 0.0, 0.0, 0.0, 0.0

            # flags
            reached, reach_info = top_block.reached(self.gripper_pos)
            lifted = top_block.lifted
            delivered = lifted and top_block.above(bottom_block)
            placed = top_block.stacked_on(bottom_block) and np.linalg.norm(self.gripper_pos - top_block.pos) < 0.2
            stacked = top_block.stacked_on(bottom_block) #and (self._stacked_flag[i] or self.gripper_pos[-1] - top_block.pos[-1] > 0.16)

            # integrate internal state
            if not self._reached_flag[i] and reached and prev_block_stacked:
                self._reached_flag[i] = True
            # reached |= self._reached_flag[i]
            if not self._lifted_flag[i] and lifted and prev_block_stacked:
                self._lifted_flag[i] = True
            # lifted |= self._lifted_flag[i] # temporary do not remember lifting
            if not self._delivered_flag[i] and delivered and prev_block_stacked:
                self._delivered_flag[i] = True
            delivered |= (self._delivered_flag[i] and lifted)
            if not self._placed_flag[i] and placed and prev_block_stacked:
                self._placed_flag[i] = True
            self._stacked_flag[i] = stacked

            # stacking reward
            num_completed_steps = 0
            if stacked and self._placed_flag[i]:
                stack_reward = self.STACKED_REWARD
                num_completed_steps = 4
            elif delivered:
                deliver_reward = self.DELIVERED_REWARD + self.DELIVERED_REWARD * self._shaped_reward(bottom_block.stack_pos,
                                                                                                     top_block.pos,
                                                                                                     prev_top_block_pos)
                num_completed_steps = 3
            elif lifted:
                lift_reward = self.LIFTED_REWARD + self.LIFTED_REWARD * self._shaped_reward(bottom_block.stack_pos,
                                                                                            top_block.pos,
                                                                                            prev_top_block_pos)
                num_completed_steps = 2
            elif reached:
                reach_reward = self.REACHED_REWARD
                num_completed_steps = 1
            else:
                reach_reward = self.REACHED_REWARD * self._shaped_reward(top_block.grasp_pos, self.gripper_pos,
                                                                         self._prev_gripper_pos)
                num_completed_steps = 0

            if prev_block_stacked:
                if self._hp.reward_density == 'dense':
                    rew += reach_reward + lift_reward + deliver_reward + stack_reward
                else:
                    rew += float(num_completed_steps) / 4

            prev_block_stacked = top_block.stacked_on(bottom_block)

            total_rew += rew
            rew_dict["rew_step{}_reach".format(i)] = np.array(reach_reward).round(3)
            rew_dict["rew_step{}_lift".format(i)] = np.array(lift_reward).round(3)
            rew_dict["rew_step{}_deliver".format(i)] = np.array(deliver_reward).round(3)
            rew_dict["rew_step{}_stack".format(i)] = np.array(stack_reward).round(3)
            rew_dict["rew_step{}_flags".format(i)] = np.array([reached, lifted, delivered, stacked], dtype=int)
            rew_dict["rc_debug{}".format(i)] = reach_info[:3]
            rew_dict["rt_debug{}".format(i)] = reach_info[3:]
            rew_dict["rew_step{}".format(i)] = np.array(rew).round(3)

            self._ep_dict['step{}'.format(i)] = rew_dict['rew_step{}_flags'.format(i)].astype(float)

        rew_dict["rew_total"] = np.array(total_rew).round(3)

        self._prev_block_pos = [copy.deepcopy(b.pos) for b in self._blocks]  # update for next round of reward comp
        self._prev_gripper_pos = copy.deepcopy(self.gripper_pos)

        return rew_dict

    def _place_objects(self):
        """
        Randomly places objects in the scene without overlap.
        """
        if self._hp.dimension == 2:
            minx = self.arena_pos[0]
            maxx = self.arena_pos[0]
        else:
            minx = -self.table_full_size[0] / 2 + self.arena_pos[0] + self._hp.block_size * 2
            maxx =  self.table_full_size[0] / 2 + self.arena_pos[0] - self._hp.block_size * 2
        miny = -self.table_full_size[1] / 2 + self.arena_pos[1] + self._hp.block_size * 2
        maxy =  self.table_full_size[1] / 2 + self.arena_pos[1] - self._hp.block_size * 2
        if self._init_block_placements is None or self._hp.rand_init_pos:
            self._init_block_placements = self._placement_gen.generate_placement(
                                              self._hp.n_blocks,
                                              side=(self._hp.block_size + self._hp.gripper_width) * 2,
                                              minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                                              fixz=not self._hp.rotate_blocks
                                          )
        block_placements = copy.deepcopy(self._init_block_placements)
        for ix, placement in enumerate(block_placements):
            block_z = self._hp.table_size[-1] + self._hp.block_size
            placement.pos = np.concatenate([placement.pos, [block_z]])
            self._blocks[ix].set_pos(placement.pos)
            self._blocks[ix].set_quat(placement.quat)

        # finish some stacking steps if the task definition has a fixed number
        # of steps required
        if self._hp.n_steps is not None:
            plan_length = len(self._task)
            for i in range(np.random.randint(plan_length - self._hp.n_steps + 1)):
                bottom_block, top_block = self._task.pop(0)

                new_pos = block_placements[bottom_block].pos
                new_pos[-1] += self._hp.block_size * 2
                new_quat = block_placements[bottom_block].quat

                self._blocks[top_block].set_pos(new_pos)
                self._blocks[top_block].set_quat(new_quat)
                block_placements[top_block].pos = new_pos
                block_placements[top_block].quat = new_quat
            self._task = self._task[:self._hp.n_steps]

        # misc
        self._last_gripper_action = None

        self._block_placements = block_placements

    def _post_action(self, action):
        reward_dict = self._get_reward()
        reward = reward_dict.rew_total # - self._prev_reward
        real_reward = reward * self._hp.reward_scale

        self._prev_reward = reward_dict.rew_total

        done = self.task_complete()

        info = reward_dict
        info.update(AttrDict(
            real_reward=real_reward.round(3),
        ))
        info.update({ k: v.round(2) for k, v in self._unflatten_block_obs(self._get_observation()).items() if not k.startswith("block")})
        for i in range(len(self._blocks)):
            info.update({ "block{}_pos".format(i): self._blocks[i].pos.round(2) })
        info.update(AttrDict(
            action=action.round(3),
            task=self._task
        ))

        return real_reward, done, info

    def get_task(self):
        return self._task

    def get_goal(self):
        """
        Get analytically-computed goal state of the current task.
        """
        # return goal if computed before in the same episode
        if self._goal is not None:
            return self._goal

        # set parameters to ensure reset doesn't change initial setup
        prev_rand_init_pos = self._hp.rand_init_pos
        prev_rand_task = self._hp.rand_task
        prev_task = copy.deepcopy(self._task)
        self._hp.rand_init_pos = False
        self._hp.rand_task = False

        # save current state
        sim_state = self.sim.get_state()

        # reset environment
        if self._t > 0:
            self.reset()

        # get block placements and virtually finish unfinished tasks in the task list
        # note that in some setups the remaining list of tasks can be partial
        # so we cannot copy initial block placements
        goal_block_placements = copy.deepcopy(self._block_placements)
        for i in range(len(self._task)):
            bottom_block, top_block = self._task.pop(0)

            new_pos = goal_block_placements[bottom_block].pos
            new_pos[-1] += self._hp.block_size * 2
            if self._hp.perturb_actions:
                new_pos[:2] += np.random.normal(0.0, self._hp.block_size / 3, 2)
            new_quat = goal_block_placements[bottom_block].quat

            self._blocks[top_block].set_pos(new_pos)
            self._blocks[top_block].set_quat(new_quat)
            goal_block_placements[top_block].pos = new_pos
            goal_block_placements[top_block].quat = new_quat

        # set robot joint positions
        robot_pos = self.sim.data.qpos[self.robot_joint_indexes]
        robot_pos[:3] = new_pos + np.array([0.0, 0.0, 0.2])
        robot_pos[-1] = 0.0
        self.sim.data.qpos[self.robot_joint_indexes] = robot_pos

        # retrieve goal observation
        goal_state = self._get_observation()

        # reset the simulator to previous state, preserving hyperparameter settings
        self.sim.set_state(sim_state)
        self.sim.forward()
        self._hp.rand_init_pos = prev_rand_init_pos
        self._hp.rand_task = prev_rand_task
        self._task = prev_task

        # save the computed episode goal state
        self._goal = goal_state

        return goal_state

    def task_complete(self):
        """Checks whether current block configuration fulfills task definition. (to verify successful execution)"""
        block_stacked = np.all([self._blocks[top_block].stacked_on(self._blocks[bottom_block])
                       for bottom_block, top_block in self._task])
        gripper_open = self.gripper_open
        gripper_lifted = self.gripper_lifted(self._blocks[self._task[-1][1]])
        task_complete = block_stacked and gripper_open and gripper_lifted
        return task_complete

    def _compute_action_penalty(self, action):
        NORM_VAL = [1., 1., 1., 0.05, 1.]       # action normalization values
        # relative actions
        rew = np.sum(np.abs(action[:4] / np.array(NORM_VAL[:4])))
        # absolute actions
        if self._last_gripper_action is not None:
            rew += np.abs((action[4] - self._last_gripper_action) / NORM_VAL[4])
        return rew

    def _position_invalid(self, pos):
        if np.abs(pos[0]) > self._hp.table_size[0] / 2: return True
        if np.abs(pos[1]) > self._hp.table_size[1] / 2: return True
        if np.abs(pos[2] - self._hp.table_size[2] - 0.5) > 0.5: return True
        return False

    def _adjust_gripper_finger_action(self, ac):
        if np.abs(ac) > 0.3 and np.abs(ac) < 0.7 and self._prev_gripper_finger_action is not None:
            return self._prev_gripper_finger_action
        self._prev_gripper_finger_action = ac

        if ac < -0.5:
            return -1.0
        elif ac > 0.5:
            return 1.0
        else:
            return 0.0

    def _pad_action(self, action):
        '''Pad omitted action spaces in the input.'''
        expected_ac_dim = 5
        if self._hp.dimension == 2:
            expected_ac_dim -= 1
        if not self._hp.allow_rotate:
            expected_ac_dim -= 1
        assert action.shape[0] == expected_ac_dim

        if self._hp.dimension == 2:
            action = np.concatenate((np.array([0]), action))
        if not self._hp.allow_rotate:
            action = np.concatenate((action[:3], np.array([0]), action[3:]))

        return action

    def _set_camera_position(self, cam_id, cam_pos):
        self.sim.model.cam_pos[cam_id] = cam_pos.copy()

    def _set_camera_rotation(self, cam_id, target_pos):
        cam_pos = self.sim.model.cam_pos[cam_id]
        forward = target_pos - cam_pos
        if forward[0] == 0 and forward[1] == 0:
            up = [0, 1, 0]
        else:
            up = [forward[0], forward[1], (forward[0]**2 + forward[1]**2) / (-forward[2] - 1e-6)]
        q = lookat_to_quat(-forward, up)
        self.sim.model.cam_quat[cam_id] = convert_quat(q, to='wxyz')

    def _get_current_subtask(self):
        """Checks which subtasks are already completed and returns first non-completed subtask."""
        for bottom_block, top_block in self._task:
            if not self._blocks[top_block].stacked_on(self._blocks[bottom_block]):
                break
        return bottom_block, top_block

    def get_episode_info(self):
        episode_info = AttrDict()

        flag_names = ['_1_reach', '_2_lift', '_3_deliver', '_4_stack']
        flag_values = [self._reached_flag, self._lifted_flag,
                       self._delivered_flag, self._stacked_flag]
        for i in range(len(flag_names)):
            episode_info.update({"block{}".format(flag_names[i]): 
                sum([int(flag_values[i][task_idx]) for task_idx in range(len(self._task))])})

        return episode_info

    @property
    def agent_params(self):
        """Parameters for agent (for demo policy)."""
        return AttrDict(
            table_size=self.table_full_size,
            block_size=self._hp.block_size,
            n_blocks=self._hp.n_blocks,
            get_task=self.get_task,
            task_complete_check=self.task_complete,
            dimension=self._hp.dimension,
            include_quat=self._hp.include_quat,
            include_vel=self._hp.include_vel,
            allow_rotate=self._hp.allow_rotate,
            max_tower_height=self._hp.task_params.max_tower_height,
        )

    def _flatten_block_obs(self, obs_dict):
        return BlockStackEnv.flatten_block_obs(obs_dict, include_quat=self._hp.include_quat, include_vel=self._hp.include_vel)

    def _unflatten_block_obs(self, obs_vector):
        return BlockStackEnv.unflatten_block_obs(obs_vector, include_quat=self._hp.include_quat, include_vel=self._hp.include_vel)

    @staticmethod
    def flatten_block_obs(obs_dict, include_quat=True, include_vel=False):
        """Flattens observation dict into vector."""
        flattened_obs = np.concatenate([
            obs_dict['gripper_pos'], obs_dict['gripper_vel'] if include_vel else np.zeros([0]),
            obs_dict['gripper_angle'][None], obs_dict['gripper_finger_pos'][None],
            obs_dict['block_pos'].flatten(), obs_dict['block_quat'].flatten()
        ])
        if not include_quat:
            flattened_obs = flattened_obs[:-len(obs_dict['block_quat'].flatten())]
        return flattened_obs

    @staticmethod
    def unflatten_block_obs(obs_vector, include_quat=True, include_vel=False):
        """Unflattens observation vector into dict."""
        n_gripper_dims = 8 if include_vel else 5
        if include_quat:
            n_blocks = (obs_vector.shape[0] - n_gripper_dims) // 7
        else:
            n_blocks = (obs_vector.shape[0] - n_gripper_dims) // 3
        if include_quat:
            block_quat = obs_vector[n_gripper_dims + n_blocks*3:n_gripper_dims + n_blocks*3 + n_blocks*4].reshape(n_blocks, 4)
        else:
            block_quat = np.array([[0.0, 0.0, 0.0, 1.0]] * n_blocks)
        return AttrDict(
            gripper_pos=obs_vector[0:3],
            gripper_vel=obs_vector[3:6] if include_vel else np.zeros([0]),
            gripper_angle=obs_vector[6] if include_vel else obs_vector[3],
            gripper_finger_pos=obs_vector[7] if include_vel else obs_vector[4],
            block_pos=obs_vector[n_gripper_dims:n_gripper_dims+n_blocks*3].reshape(n_blocks, 3),
            block_quat=block_quat
        )

    def obs2qpos(self, obs):
        """Converts observation to simulator qpos representation. Used for restoration of env state."""
        if not type(obs) == AttrDict:
            obs = self._unflatten_block_obs(obs)

        qpos = []

        # env qpos layout:
        # 'gripper_pos', 'gripper_angle', 'gripper_finger_pos', <- 5
        # 'neg_gripper_finger_pos', <- need to be added
        # ['obj#_pos', 'obj#_quat'] x n
        qpos.append(obs.gripper_pos)
        qpos.append(np.array([obs.gripper_angle]))
        qpos.append(np.array([obs.gripper_finger_pos, -obs.gripper_finger_pos]))
        for i in range(obs.block_pos.shape[0]):
            qpos.append(obs.block_pos[i])
            qpos.append(obs.block_quat[i])
        qpos = np.concatenate(qpos)

        assert len(qpos) == len(BlockStackEnv.flatten_block_obs(obs, True, False)) + 1

        return qpos

    @property
    def gripper_open(self):
        return np.abs(self.sim.data.qpos[self.gripper_joint_indexes][0]) > 1e-2

    @property
    def gripper_pos(self):
        return self.sim.data.qpos[self.robot_joint_indexes][:3]

    @property
    def gripper_vel(self):
        return self.sim.data.qvel[self.robot_joint_indexes][:3]

    @property
    def gripper_angle(self):
        return self.sim.data.qpos[self.robot_joint_indexes][-1]

    @property
    def gripper_finger_pos(self):
        return np.abs(self.sim.data.qpos[self.gripper_joint_indexes][0])

    @property
    def gripper_finger_poses(self):
        return [self.sim.data.qpos[self.gripper_joint_indexes][0],
                self.sim.data.qpos[self.gripper_joint_indexes][1]]

    @property
    def gripper_finger_dist(self):
        return np.abs(self.sim.data.qpos[self.gripper_joint_indexes][0]
                      - self.sim.data.qpos[self.gripper_joint_indexes][1])

    @property
    def max_height(self):
        """Maximum permissable height of gripper."""
        return 0.7

    def gripper_lifted(self, block):
        """Indicates whether gripper is lifted above block."""
        return self.sim.data.qpos[self.robot_joint_indexes][2] - block.pos[2] > 0.2

    def _shaped_reward(self, target, value, prev_value=None):
        """Tanh-shaped reward function."""
        if self._hp.relative_shaped_reward:
            return min(1, self.RELATIVE_REWARD_SCALE * (np.linalg.norm(target - prev_value)
                                                        - np.linalg.norm(target - value)))
        return 1 - np.tanh(self.DENSE_REWARD_SCALING_FACTOR * np.linalg.norm(target - value)) ** 2

    def _load_model(self):
        """
        Loads an xml model and puts it in self.model.
        """
        super()._load_model()

        # creating the model
        self.model = MujocoWorldBase()

        # adding robot
        self.mujoco_robot = GripperOnlyRobot()
        self.mujoco_robot.set_base_xpos(self.robot_pos)

        # add free-form gripper
        self.gripper = WideRangeGripper()
        self.gripper.hide_visualization()
        self.mujoco_robot.add_gripper("hand", self.gripper)
        self.model.merge(self.mujoco_robot)

        # adding table
        self.mujoco_arena = TableArena(table_full_size=self.table_full_size)
        self.mujoco_arena.set_origin(self.arena_pos)
        self.model.merge(self.mujoco_arena)

        # add invisible boundary walls
        if self._hp.add_boundary_walls:
            TOP_OFFSET = 0.2        # not sure why this is necessary, but robot seems to have invisible top-part?
            for name, size, pos in zip(['wall_left', 'wall_right', 'wall_top'],
                    [[self.table_full_size[0]/2, 0.01, self.max_height/2 + TOP_OFFSET/2],
                     [self.table_full_size[0]/2, 0.01, self.max_height/2 + TOP_OFFSET/2],
                     [self.table_full_size[0]/2, self.table_full_size[1]/2, 0.01]],
                    [[0, -self.table_full_size[1] / 2, self.max_height / 2 + self._hp.table_size[2] + TOP_OFFSET/2],
                     [0, self.table_full_size[1] / 2, self.max_height / 2 + self._hp.table_size[2] + TOP_OFFSET/2],
                     [0, 0, self.max_height + self._hp.table_size[2] + TOP_OFFSET]]):
                wall_body = new_body(name, pos=pos)
                geom_obj = new_geom("box", size=size, rgba=(0,0,0,0), group=1, contype="2", conaffinity="2",)
                wall_body.append(geom_obj)
                self.model.worldbody.append(wall_body)

        # adding objects
        self._blocks = []
        for ix in range(self._hp.n_blocks):
            item_color = color_dict
            item_name = 'obj{}'.format(ix)

            block_color = np.array(color_dict[self._hp.block_color])
            block_size = [self._hp.block_size] * 3
            mujoco_obj = NumberedBoxObject(size=block_size,
                                           rgba=block_color.tolist(),
                                           friction=self._hp.friction,
                                           number=ix if self._hp.number_blocks else None)
            self.model.merge_asset(mujoco_obj)

            mjcf_obj = mujoco_obj.get_collision(name=item_name, site=True)
            mjcf_obj.append(new_joint(name=item_name, type="free"))
            self.model.worldbody.append(mjcf_obj)

            self._blocks.append(Block(AttrDict({
                "env": self,
                "name": item_name,
                "size": block_size,
                "mujoco_obj": mujoco_obj,
                "mjcf_obj": mjcf_obj
            })))

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # robot joint indices
        self.robot_joints = list(self.mujoco_robot.joints)
        self.robot_joint_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]

        # gripper joint indices
        self.gripper_joints = list(self.gripper.joints)
        self.gripper_joint_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
        ]

        # block and gripper body and geom ids
        self.block_body_ids = [
            self.sim.model.body_name2id(block.name) for block in self._blocks
        ]
        self.block_geom_ids = [
            self.sim.model.geom_name2id(block.name) for block in self._blocks
        ]
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")


class NoOrderBlockStackEnv(BlockStackEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_task = self._fixed_task + [(0,0)]
        self._reset_internal()

    def _get_reward(self):
        """Compute reward for stacking blocks without order."""
        rew_dict = AttrDict()
        total_rew = 0.0

        for i, block in enumerate(self._blocks):
            rew = 0.0

            reach_reward, lift_reward, deliver_reward, stack_reward = 0.0, 0.0, 0.0, 0.0

            # flags
            lifted = block.lifted
            stacked = any([block.stacked_on_loose(b) for b in self._blocks if block.name != b.name])

            # integrate internal state
            if not self._lifted_flag[i] and lifted:
                self._lifted_flag[i] = True
            if not self._stacked_flag[i] and stacked:
                self._stacked_flag[i] = stacked
            self._stacked_final_flag[i] = stacked

            # stacking reward
            if stacked:
                stack_reward = self.STACKED_REWARD
            elif lifted:
                lift_reward = self.LIFTED_REWARD

            rew += reach_reward + lift_reward + deliver_reward + stack_reward

            total_rew += rew
            rew_dict["rew_step{}_lift".format(i)] = np.array(lift_reward).round(3)
            rew_dict["rew_step{}_stack".format(i)] = np.array(stack_reward).round(3)
            rew_dict["rew_step{}_flags".format(i)] = np.array([lifted, stacked], dtype=int)
            rew_dict["rew_step{}".format(i)] = np.array(rew).round(3)

            self._ep_dict['step{}'.format(i)] = rew_dict['rew_step{}_flags'.format(i)].astype(float)

        rew_dict["rew_total"] = np.array(total_rew).round(3)

        self._prev_block_pos = [copy.deepcopy(b.pos) for b in self._blocks]  # update for next round of reward comp
        self._prev_gripper_pos = copy.deepcopy(self.gripper_pos)

        return rew_dict

    def get_episode_info(self):
        episode_info = AttrDict()

        flag_names = ['_1_grasped', '_2_lift', '_4_stack', '_5_stack_final']
        flag_values = [self._grasped_flag, self._lifted_flag, self._stacked_flag, self._stacked_final_flag]
        for i in range(len(flag_names)):
            episode_info.update({"block{}".format(flag_names[i]):
                sum([int(flag_values[i][task_idx]) for task_idx in range(len(self._task))])})

        return episode_info

    def _reset_internal(self, keep_sim_object=False):
        super()._reset_internal(keep_sim_object)
        # reset reward internal states
        self._reached_flag = [False] * len(self._blocks)
        self._grasped_flag = [False] * len(self._blocks)
        self._lifted_flag = [False] * len(self._blocks)
        self._delivered_flag = [False] * len(self._blocks)
        self._placed_flag = [False] * len(self._blocks)
        self._stacked_flag = [False] * len(self._blocks)
        self._stacked_final_flag = [False] * len(self._blocks)


class HighStackBlockStackEnv(NoOrderBlockStackEnv):
    """Reward is proportional to highest stacked tower."""
    REWARD_SCALE = 50
    LIFTED_REWARD = 1.0
    LIFTED_ABOVE_REWARD = 2.0
    ROTATION_PENALTY = 1

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'restrict_upright': False,      # if True, requires block to be stacked upright to get reward
            'restrict_grasped': True,       # if True, requires block to be grasped before stacking to get reward
            'rotation_penalty': False,      # if True, adds penalty for rotated blocks
        }))

    def _reset_internal(self, keep_sim_object=False):
        super()._reset_internal(keep_sim_object)
        self._final_height = 0.

    def get_episode_info(self):
        ep_info = super().get_episode_info()
        ep_info.final_height = self._final_height
        return ep_info

    def _get_reward(self):
        """Compute reward for stacking blocks without order."""
        rew_dict = AttrDict()

        max_height, total_rew = 0., 0.
        heights, supported_heights = np.zeros(len(self._blocks)), np.zeros(len(self._blocks))
        for i, block in enumerate(self._blocks):
            height = block.dist_lifted
            heights[i] = height

            # set flags
            if not self._grasped_flag[i]:
                self._grasped_flag[i] = block.grasped(self.gripper_pos, self.gripper_finger_dist, self.gripper_finger_poses)
            if not self._lifted_flag[i]:
                self._lifted_flag[i] = (not self._hp.restrict_grasped or self._grasped_flag[i]) and \
                        (not self._hp.restrict_upright or block.upright) and block.lifted
            if not self._delivered_flag[i]:
                self._delivered_flag[i] = (not self._hp.restrict_grasped or self._grasped_flag[i]) \
                        and (not self._hp.restrict_upright or block.upright) \
                        and any([block.above(b) for b in self._blocks if b.name != block.name])

            # compute reward
            if (not self._hp.restrict_grasped or self._grasped_flag[i]) and \
                    (not self._hp.restrict_upright or block.upright) and \
                    self._has_support(block, [b for b in self._blocks if block.name != b.name]):
                self._stacked_flag[i] = True
                supported_heights[i] = height
                if height > max_height:
                    max_height = height
            if self._delivered_flag[i]:
                total_rew += self.LIFTED_ABOVE_REWARD
            elif self._lifted_flag[i]:
                total_rew += self.LIFTED_REWARD
        self._final_height = max_height / (2*self._hp.block_size)

        total_rew += max_height * self.REWARD_SCALE

        if self._hp.rotation_penalty:
            # add per-step penalty for each rotated block
            rot_penalty = sum([self.ROTATION_PENALTY if not b.upright else 0 for b in self._blocks])
            total_rew -= rot_penalty
            rew_dict["rot_penalty"] = np.array(rot_penalty).round(3)

        rew_dict["heights"] = heights.round(3)
        rew_dict["sup_heights"] = supported_heights.round(3)
        rew_dict["rew_total"] = np.array(total_rew).round(3)
        rew_dict["max_height"] = np.array(self._final_height).round(3)
        #rew_dict["z_ang"] = np.array([b.z_angle * 180 / np.pi for b in self._blocks]).round(1)
        rew_dict["grasped_1"] = np.array(self._grasped_flag[:5])
        rew_dict["grasped_2"] = np.array(self._grasped_flag[5:])
        rew_dict["lifted_1"] = np.array(self._lifted_flag[:5])
        rew_dict["lifted_2"] = np.array(self._lifted_flag[5:])
        rew_dict["gripper_finger_dist"] = np.array(self.gripper_finger_dist).round(3)


        self._prev_block_pos = [copy.deepcopy(b.pos) for b in self._blocks]  # update for next round of reward comp
        self._prev_gripper_pos = copy.deepcopy(self.gripper_pos)

        return rew_dict

    def _has_support(self, block, others):
        return not block.lifted or any([block.stacked_on_loose(b) and self._has_support(b, [bb for bb in others if b.name != bb.name])
                    for b in others])


class SparseHighStackBlockStackEnv(NoOrderBlockStackEnv):
    """Simple reward function that just rewards the highest stacked tower."""
    REWARD_SCALE = 1.0

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'restrict_upright': True,       # if True, requires block to be stacked upright to get reward
            'restrict_grasped': False,      # if True, requires block to be grasped before stacking to get reward
            'rotation_penalty': False,      # if True, adds penalty for rotated blocks
        }))

    def _reset_internal(self, keep_sim_object=False):
        super()._reset_internal(keep_sim_object)
        self._final_height = 0.

    def get_episode_info(self):
        ep_info = super().get_episode_info()
        ep_info.final_height = self._final_height
        return ep_info

    def _get_reward(self):
        """Compute reward for stacking blocks without order."""
        rew_dict = AttrDict()

        max_height = 0.
        heights, supported_heights = np.zeros(len(self._blocks)), np.zeros(len(self._blocks))
        for i, block in enumerate(self._blocks):
            height = block.dist_lifted
            heights[i] = height

            # set flags
            if not self._grasped_flag[i]:
                self._grasped_flag[i] = block.grasped(self.gripper_pos, self.gripper_finger_dist,
                                                      self.gripper_finger_poses)
            if not self._lifted_flag[i]:
                self._lifted_flag[i] = (not self._hp.restrict_grasped or self._grasped_flag[i]) and \
                                       (not self._hp.restrict_upright or block.upright) and block.lifted
            if not self._delivered_flag[i]:
                self._delivered_flag[i] = (not self._hp.restrict_grasped or self._grasped_flag[i]) \
                                          and (not self._hp.restrict_upright or block.upright) \
                                          and any([block.above(b) for b in self._blocks if b.name != block.name])

            # compute reward
            if (not self._hp.restrict_grasped or self._grasped_flag[i]) and \
                    (not self._hp.restrict_upright or block.upright) and \
                    self._has_support(block, [b for b in self._blocks if block.name != b.name]):
                self._stacked_flag[i] = True
                supported_heights[i] = height
                if height > max_height:
                    max_height = height
        self._final_height = max_height / (2 * self._hp.block_size)

        total_rew = max_height * self.REWARD_SCALE

        rew_dict["heights"] = heights.round(3)
        rew_dict["sup_heights"] = supported_heights.round(3)
        rew_dict["rew_total"] = np.array(total_rew).round(3)
        rew_dict["max_height"] = np.array(self._final_height).round(3)

        self._prev_block_pos = [copy.deepcopy(b.pos) for b in self._blocks]  # update for next round of reward comp
        self._prev_gripper_pos = copy.deepcopy(self.gripper_pos)

        return rew_dict

    def _has_support(self, block, others):
        return not block.lifted or any([block.stacked_on_loose(b) and self._has_support(b, [bb for bb in others if b.name != bb.name])
                    for b in others])
