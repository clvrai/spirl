from spirl.data.block_stacking.src.robosuite.models.grippers.two_finger_gripper import TwoFingerGripper


class WideRangeGripper(TwoFingerGripper):
    def __init__(self):
        super().__init__()
        l_gripper = self.worldbody.find(".//joint[@name='r_gripper_l_finger_joint']")
        l_gripper.set("range", "-0.020833 0.08")
        l_actuator = self.actuator.find(".//position[@joint='r_gripper_l_finger_joint']")
        l_actuator.set("ctrlrange", "-0.020833 0.08")
        l_actuator.set("gear", "2")
        r_gripper = self.worldbody.find(".//joint[@name='r_gripper_r_finger_joint']")
        r_gripper.set("range", "-0.08 0.020833")
        r_actuator = self.actuator.find(".//position[@joint='r_gripper_r_finger_joint']")
        r_actuator.set("ctrlrange", "-0.08 0.020833")
        r_actuator.set("gear", "2")

        for geom_name in ['l_finger_g1', 'r_finger_g1']:
            geom = self.worldbody.find(".//geom[@name='{}']".format(geom_name))
            geom.set("conaffinity", "0")

        for geom_name in ['l_fingertip_g0', 'r_fingertip_g0']:
            geom = self.worldbody.find(".//geom[@name='{}']".format(geom_name))
            geom.set("friction", "30 0.005 0.0001")
