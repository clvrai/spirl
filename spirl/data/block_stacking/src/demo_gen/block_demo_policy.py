import numpy as np
from collections import deque
import copy

from spirl.utils.general_utils import AttrDict, split_along_axis
from spirl.data.block_stacking.src.utils.utils import quat2euler
from spirl.data.block_stacking.src.block_stacking_env import BlockStackEnv


class BlockStackDemoPolicy:
    """Follows plan on given env."""
    GRASP_OFFSET = 0.08        # offset between robot pos and block pos for grasping
    PICK_OFFSET = 0.14         # additional vertical offset btw robot and block for placing
    PLACE_OFFSET = 0.17       # additional vertical offset btw robot and block for placing
    ACT_RANGE = [0.05, 0.05, 0.05, np.pi/10, 0.5]  # maximum action scale for each action dimension
    GRAVITY_SUPPORT = 0.01     # z dimension action when noop to prevent robot from falling
    GRIPPER_OPEN = 1.
    GRIPPER_CLOSED = 0.
    MULTIPLIER = 20.
    EPS = 0.01

    def __init__(self, env_params):
        """
        :param hl_plan: list of HL index tuples indicating which block should get stacked (e.g. [(1,2), (3,5)])
        """
        # TODO consider whether to make task/hl_plan a proper class with transition subclass (to make reuse for kitchen easier)
        self.env_params = env_params
        self.lift_height = env_params.table_size[-1] + env_params.block_size * 2 * env_params.max_tower_height + 0.2
        self.block_height = env_params.block_size * 2

        self._hl_plan = None
        self._hl_plan_to_run = deque()
        self._action_plan = None
        self._u_obs = None  # this stores env state when planning action sequence
        self._update_robot_state = True

    def reset(self):
        self._hl_plan = self.env_params.get_task()
        self._action_plan = None
        self._hl_plan_to_run = deque(self._hl_plan)
        self._u_obs = None

    def act(self, obs):
        if self.execution_finished: # should not call 'act' if execution is already finished
            return None
        self._u_obs = BlockUnflattenWrapper(BlockStackEnv.unflatten_block_obs(copy.deepcopy(obs),
                                                                              include_quat=self.env_params.include_quat,
                                                                              include_vel=self.env_params.include_vel))
        while True:
            if self._action_plan is None:
                if not self._hl_plan_to_run:
                    self._action_plan = None
                    ac = np.zeros(5,)
                    break
                # generate new action plan
                self._action_plan = self._plan_actions()
            try:
                ac = next(self._action_plan)
                break
            except (StopIteration, IndexError): # generator exhausted
                self._action_plan = None
        ac = self._post_process(ac)
        return ac

    @property
    def execution_finished(self):
        """Checks whether the plan execution has been finished."""
        return self._action_plan is None and not self._hl_plan_to_run

    def _plan_actions(self):
        """Plans LL actions given HL action plan and current env state."""
        # generate pick-place plan for one stacking subtask
        bottom_block, top_block = self._hl_plan_to_run.popleft()
        raw_plan = self._pick_place(bottom_block, top_block)

        for ac in split_along_axis(raw_plan, axis=0):
            yield ac

    def _pick_place(self, bottom_block, top_block):
        """Plans action sequence for pick&place of single block."""
        action_plan = []

        # pick up block
        pick_target_pos = self._get_pick_target(top_block)
        top_block_quat = self._u_obs.block_quat(top_block)
        action_plan.append(self._move_to(pick_target_pos, top_block_quat, self.GRIPPER_OPEN)[0])
        action_plan.append(self._grasp())

        # place block
        place_target_pos = self._get_place_target(bottom_block)
        bottom_block_quat = self._u_obs.block_quat(bottom_block)
        action_plan.append(self._move_to(place_target_pos, bottom_block_quat, self.GRIPPER_CLOSED)[0])
        action_plan.append(self._place())

        return np.concatenate(action_plan)

    def _get_pick_target(self, block):
        block_pos = self._u_obs.block_pos(block)
        block_pos[2] += self.PICK_OFFSET
        return block_pos

    def _get_place_target(self, block):
        block_pos = self._u_obs.block_pos(block)
        block_pos[2] += self.PLACE_OFFSET
        return block_pos

    def _move_to(self, target_block_pos, target_block_quat, gripper, waypoints=None):
        """
        Plans action sequence for moving robot arm to block.
        :param gripper: indicates whether gripper should be ['open', 'closed'] during execution
        :param waypoints: (optional) list of precomputed waypoints
        """
        block_angle = quat2euler(*target_block_quat)[0]  # assume single-axis rotation
        robot_pos, robot_angle = self._u_obs.gripper_pos, self._u_obs.gripper_angle
        if waypoints is None:
            waypoints = [
                [robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self._u_obs.gripper_finger_pos],
                [robot_pos[0], robot_pos[1], self.lift_height, robot_angle, gripper],
                [target_block_pos[0], target_block_pos[1], self.lift_height, robot_angle, gripper],
                [target_block_pos[0], target_block_pos[1], target_block_pos[2] + self.GRASP_OFFSET, block_angle, gripper],
            ]

            # add disturbed subgoals in between waypoints for better state coverage
            subgoals = [
                self._sample_disturbed_subgoal(robot_pos,
                                               [robot_pos[0], robot_pos[1], self.lift_height])
                                                + [robot_angle, gripper],
                self._sample_disturbed_subgoal([robot_pos[0], robot_pos[1], self.lift_height],
                                               [target_block_pos[0], target_block_pos[1], self.lift_height])
                                                + [robot_angle, gripper],
                self._sample_disturbed_subgoal([target_block_pos[0], target_block_pos[1], self.lift_height],
                                               [target_block_pos[0], target_block_pos[1], target_block_pos[2] + self.GRASP_OFFSET])
                                                + [block_angle, gripper],
            ]

            # assemble final waypoint list
            waypoints = [waypoints[0], subgoals[0], waypoints[1], subgoals[1], waypoints[2], subgoals[2], waypoints[3]]
        else:
            waypoints = [[robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self._u_obs.gripper_finger_pos]] \
                            + waypoints

        if self._update_robot_state:
            self._u_obs.gripper_pos, self._u_obs.gripper_angle, self._u_obs.gripper_finger_pos = \
                np.array(waypoints[-1][:3]), waypoints[-1][3], gripper      # update robot state
        return self._waypoints2plan(waypoints, absolute_dims=[-1]), waypoints[1:]

    def _grasp(self):
        """Moves robot GRASP-offset down, closes gripper, moves GRASP-offset up."""
        robot_pos, robot_angle = self._u_obs.gripper_pos, self._u_obs.gripper_angle
        waypoints = [
            [robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self.GRIPPER_OPEN],
            [robot_pos[0], robot_pos[1], robot_pos[2] - self.GRASP_OFFSET, robot_angle, self.GRIPPER_OPEN],
            [robot_pos[0], robot_pos[1], robot_pos[2] - self.GRASP_OFFSET, robot_angle, self.GRIPPER_CLOSED]]
        waypoints += [waypoints[-1]] * 3 # noop
        waypoints += [[robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self.GRIPPER_CLOSED]]
        if self._update_robot_state:
            self._u_obs.gripper_finger_pos = self.GRIPPER_CLOSED    # update robot state
        return self._waypoints2plan(waypoints, absolute_dims=[-1])

    def _place(self):
        """Moves robot GRASP-offset down, opens gripper, moves GRASP-offset up."""
        robot_pos, robot_angle = self._u_obs.gripper_pos, self._u_obs.gripper_angle
        waypoints = [
            [robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self.GRIPPER_CLOSED],
            [robot_pos[0], robot_pos[1], robot_pos[2] - self.GRASP_OFFSET, robot_angle, self.GRIPPER_CLOSED],
            [robot_pos[0], robot_pos[1], robot_pos[2] - self.GRASP_OFFSET, robot_angle, self.GRIPPER_OPEN],
            [robot_pos[0], robot_pos[1], robot_pos[2], robot_angle, self.GRIPPER_OPEN],
            [robot_pos[0], robot_pos[1], self.lift_height, robot_angle, self.GRIPPER_OPEN]
        ]
        if self._update_robot_state:
            self._u_obs.gripper_finger_pos = self.GRIPPER_OPEN  # update robot state
        return self._waypoints2plan(waypoints, absolute_dims=[-1])

    def _waypoints2plan(self, waypoints, absolute_dims=None):
        plan = np.concatenate([self._interpolate(waypoints[i], waypoints[i+1], absolute_dims)
                               for i in range(len(waypoints) - 1)])
        return plan

    def _interpolate(self, start, goal, absolute_dims=None):
        """
        Interpolates between start and goal linearly while taking max_actions into account.
        Since action effect is smaller than actual action scale we need a multiplier to treat the distance farther than the actual one.
        :param absolute_dims: list of dimensions for which action will be set to goal state.
        """
        diff = np.array(goal) - np.array(start)
        n_steps = int(np.max(np.ceil(np.divide(np.abs(diff), np.array(self.ACT_RANGE)))))
        for dim in absolute_dims if absolute_dims is not None else []:
            diff[dim] = goal[dim] * n_steps   # hack to make dims action values absolute
        if n_steps > 0:
            actions = [diff / n_steps for _ in range(n_steps)]
            return actions
        else:
            return np.zeros([0, diff.shape[-1]])

    def _post_process(self, ac):
        # scale action
        ac[:3] *= self.MULTIPLIER      # scale lateral actions to make them reach the target states

        # add gravity support for noop
        if np.sum(ac[:-1]) == 0:
            ac[2] += self.GRAVITY_SUPPORT

        # crop action dimensions according to env params
        if not self.env_params.allow_rotate:
            ac = np.concatenate([ac[:3], ac[4:]])
        if self.env_params.dimension == 2:
            ac = ac[1:]

        return ac

    def _sample_disturbed_subgoal(self, start_pos, goal_pos, max_displacement_ratio=0.2):
        """Samples a subgoal with some offset to the direct connection line."""
        start_pos, goal_pos = np.array(start_pos), np.array(goal_pos)
        diff = goal_pos - start_pos

        # generate unit vector that's orthogonal to diff
        noise = np.asarray([diff[0], diff[2], -diff[1]])
        noise /= np.linalg.norm(noise)  # normalize it

        # sample random offset along connection line + random length
        length = (np.random.rand() * 2 * max_displacement_ratio - max_displacement_ratio) * np.linalg.norm(diff)
        offset = (np.random.rand() * 0.6 + 0.2) * diff

        # compute subgoal position
        subgoal_pos = start_pos + offset + length * noise
        return [coord for coord in subgoal_pos]



class ClosedLoopBlockStackDemoPolicy(BlockStackDemoPolicy):
    PICK_OFFSET = 0.11

    def __init__(self, env_params):
        super().__init__(env_params)
        self._update_robot_state = False

    def _plan_actions(self):
        # generate pick-place plan for one stacking subtask
        bottom_block, top_block = self._hl_plan_to_run.popleft()
        top_block_init_pos = self._u_obs.block_pos(top_block)

        waypoints = None
        while not self._lifted(top_block):
            while not self._reached(self._get_pick_target(top_block)):
                pick_target_pos = self._get_pick_target(top_block)
                top_block_quat = self._u_obs.block_quat(top_block)
                actions, waypoints = self._move_to(pick_target_pos, top_block_quat, self.GRIPPER_OPEN, waypoints)
                if self._reached_waypoint(waypoints[0]) and len(waypoints) > 1:
                    waypoints = waypoints[1:]
                if len(actions) > 0:
                    yield actions[0]
                else:
                    break

            grasp_plan = split_along_axis(self._grasp(), axis=0)
            for i, action in enumerate(grasp_plan):
                yield action

        waypoints = None
        while not self._reached(self._get_place_target(bottom_block)):
            place_target_pos = self._get_place_target(bottom_block)
            bottom_block_quat = self._u_obs.block_quat(bottom_block)
            actions, waypoints = self._move_to(place_target_pos, bottom_block_quat, self.GRIPPER_CLOSED, waypoints)
            if self._reached_waypoint(waypoints[0]) and len(waypoints) > 1:
                waypoints = waypoints[1:]
            if len(actions) > 0:
                yield actions[0]
            else:
                break

        while not self._stacked(top_block, bottom_block):
            for action in split_along_axis(self._place(), axis=0):
                yield action

    def _lifted(self, top_block):
        top_block_pos = self._u_obs.block_pos(top_block)
        gripper_pos = self._u_obs.gripper_pos

        lifted = True

        x_dist = np.abs(gripper_pos[0] - top_block_pos[0])
        lifted &= x_dist < self.env_params.block_size

        y_dist = np.abs(gripper_pos[1] - top_block_pos[1])
        lifted &= y_dist < self.env_params.block_size

        z_vec = gripper_pos[-1] - top_block_pos[-1]
        lifted &= z_vec < 0.14
        lifted &= z_vec > 0.08

        return lifted

    def _stacked(self, top_block, bottom_block):
        top_pos = self._u_obs.block_pos(top_block)
        bottom_pos = self._u_obs.block_pos(bottom_block)
        x_dist = np.linalg.norm(top_pos[0] - bottom_pos[0])
        y_dist = np.linalg.norm(top_pos[0] - bottom_pos[0])
        x_dist_correct = x_dist < self.env_params.block_size
        y_dist_correct = y_dist < self.env_params.block_size

        z_vec = top_pos[2] - bottom_pos[2]
        z_vec_correct = np.abs(z_vec - 2 * self.env_params.block_size) < 0.005

        return x_dist_correct and y_dist_correct and z_vec_correct

    def _reached(self, pos):
        target_pos = pos
        target_pos[2] += self.GRASP_OFFSET
        return np.linalg.norm(pos - self._u_obs.gripper_pos) < self.EPS

    def _reached_waypoint(self, waypoint):
        return np.linalg.norm(np.array(waypoint[:3]) - self._u_obs.gripper_pos) < self.EPS


class BlockUnflattenWrapper(AttrDict):
    def block_pos(self, idx):
        return list(self['block_pos'][idx])

    def block_quat(self, idx):
        return list(self['block_quat'][idx])

    def set_block_pos(self, idx, val):
        self['block_pos'][idx] = val

    def set_block_quat(self, idx, val):
        self['block_quat'][idx] = val


if __name__ == "__main__":
    from spirl.data.block_stacking.src.block_task_generator import SingleTowerBlockTaskGenerator
    obs = AttrDict(
        block_pos=np.random.rand(4*3),
        block_quat=np.random.rand(4*4),
        gripper_pos=np.random.rand(3),
        gripper_angle=np.random.rand(),
        gripper_finger_pos=np.random.rand(),
    )
    task_gen = SingleTowerBlockTaskGenerator({}, 4)
    task = task_gen.sample()
    policy = BlockStackDemoPolicy(task)
    print(policy.act(obs))
    # print(policy._plan_actions(obs))


