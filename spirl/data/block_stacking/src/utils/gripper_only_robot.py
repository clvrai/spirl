import os
import numpy as np
from spirl.data.block_stacking.src.robosuite.models.robots.robot import Robot
from spirl.data.block_stacking.src.robosuite.utils.mjcf_utils import array_to_string


class GripperOnlyRobot(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(self):
        super().__init__(os.path.join(os.getcwd(), "spirl/data/block_stacking/assets/gripper_only_robot.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 4

    @property
    def joints(self):
        return ["slide_x", "slide_y", "slide_z", "rotate_z"]

    @property
    def init_qpos(self):
        return np.array([0, 0, 1.2, 0])
