import numpy as np
import math
from collections import deque


class Block(object):
    """Block object."""
    def __init__(self, config):
        self._env = config.env
        self._name = config.name
        self._size = config.size    # this is really more something like a radius, i.e. half the length of the side
        self._mujoco_obj = config.mujoco_obj
        self._mjcf_obj = config.mjcf_obj
        self._starting_height = None
        self.reset()

    def reset(self):
        self._grasp_buffer = deque([True] * 5, maxlen=5)   # stores whether block been grasped for N consecutive steps

    def stacked_on(self, other):
        """Checks whether current block is placed on other block."""
        x_dist = np.linalg.norm(self.pos[0] - other.pos[0])
        y_dist = np.linalg.norm(self.pos[1] - other.pos[1])
        x_dist_correct = x_dist < other.size[0]
        y_dist_correct = y_dist < other.size[1]

        z_vec = self.pos[-1] - other.pos[-1]
        z_vec_correct = np.abs(z_vec - (self.size[-1] + other.size[-1])) < 0.005

        return x_dist_correct and y_dist_correct and z_vec_correct

    def stacked_on_loose(self, other):
        """Checks whether current block is placed on other block (with wider margin)."""
        x_dist = np.linalg.norm(self.pos[0] - other.pos[0])
        y_dist = np.linalg.norm(self.pos[1] - other.pos[1])
        x_dist_correct = x_dist < 2*other.size[0]
        y_dist_correct = y_dist < 2*other.size[1]

        z_vec = self.pos[-1] - other.pos[-1]
        z_vec_correct = self.size[-1] < z_vec < self.size[-1] + other.size[-1] + 0.005

        return x_dist_correct and y_dist_correct and z_vec_correct

    def above(self, other):
        """Checks whether current block is above other block."""
        x_dist = np.linalg.norm(self.pos[0] - other.pos[0])
        y_dist = np.linalg.norm(self.pos[1] - other.pos[1])
        x_dist_correct = x_dist < other.size[0]
        y_dist_correct = y_dist < other.size[1]

        z_vec = self.pos[-1] - other.pos[-1]
        z_vec_correct = z_vec > self.size[-1] + other.size[-1] - 0.005

        return x_dist_correct and y_dist_correct and z_vec_correct

    def grasped(self, gripper_pos, gripper_finger_dist, gripper_finger_poses):
        # gripper over block & block lifted & gripper closed & fingers around block
        is_grasped = self.reached(gripper_pos) \
                     and self.lifted and 0.03 <= gripper_finger_dist <= 0.05 \
                     and gripper_finger_poses[1] + gripper_pos[1] <= self.pos[1] - 0.01 \
                     and gripper_finger_poses[0] + gripper_pos[1] >= self.pos[1] + 0.01
        self._grasp_buffer.append(True if is_grasped else False)
        return all(list(self._grasp_buffer))    # return true if block was grasped for N consecutive time steps

    @property
    def lifted(self):
        return self.pos[-1] > self.starting_height + (math.sqrt(3) - 1.0) * self.size[-1]

    @property
    def dropped_off_table(self):
        return self.pos[-1] < self.starting_height - self.size[-1]

    @property
    def upright(self):
        # compute angle with z-axis, return true if smaller 45 degree
        return abs(self.z_angle) <= 45 * np.pi / 180

    @property
    def z_angle(self):
        # compute angle with z-axis
        q = self.quat
        z_axis = [2 * q[1] * q[3] + 2 * q[2] * q[0], 2 * q[2] * q[3] - 2 * q[1] * q[0],
                  1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]
        return math.acos(np.dot(np.array(z_axis), np.array([0, 0, 1])))

    def reached(self, gripper_pos):
        x_dist = np.linalg.norm(self.pos[0] - gripper_pos[0])
        y_dist = np.linalg.norm(self.pos[1] - gripper_pos[1])
        x_dist_correct = x_dist < 2 * self.size[0]
        y_dist_correct = y_dist < 2 * self.size[1]

        z_vec = gripper_pos[-1] - self.pos[-1]
        z_vec_correct = z_vec > 0.16 and z_vec < 0.22

        reached = x_dist_correct and y_dist_correct and z_vec_correct
        info = np.array([x_dist, y_dist, z_vec, 2 * self.size[0], 2 * self.size[1], 0.22]).round(2)
        return reached, info

    def set_pos(self, pos):
        ix_start, _ = self._env.sim.model.get_joint_qpos_addr(self._name)
        self._env.sim.data.qpos[ix_start:ix_start+3] = pos

    def set_quat(self, quat):
        ix_start, ix_end = self._env.sim.model.get_joint_qpos_addr(self._name)
        self._env.sim.data.qpos[ix_start+3:ix_end] = quat

    @property
    def pos(self):
        ix_start, _ = self._env.sim.model.get_joint_qpos_addr(self._name)
        return self._env.sim.data.qpos[ix_start:ix_start+3]

    @property
    def grasp_pos(self):
        """Target position from which block can be grasped."""
        target_pos = self.pos.copy()
        target_pos[-1] += 0.16
        return target_pos

    @property
    def stack_pos(self):
        """Target position where stacked block should be placed."""
        target_pos = self.pos.copy()
        target_pos[-1] += 3 * self.size[-1] # this assumes that all blocks have the same size
        return target_pos

    @property
    def starting_height(self):
        if self._starting_height is None:
            self._starting_height = self.pos[-1].copy()
        return self._starting_height

    @property
    def quat(self):
        ix_start, ix_end = self._env.sim.model.get_joint_qpos_addr(self._name)
        return self._env.sim.data.qpos[ix_start+3:ix_end]

    @property
    def dist_lifted(self):
        return max(0, self.pos[-1] - self.starting_height)

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
