import numpy as np

from spirl.components.logger import Logger
from spirl.models.skill_prior_mdl import SkillSpaceLogger
from spirl.data.block_stacking.src.block_stacking_env import BlockStackEnv
from spirl.utils.general_utils import AttrDict
from spirl.utils.vis_utils import add_caption_to_img
from spirl.data.block_stacking.src.block_task_generator import FixedSizeSingleTowerBlockTaskGenerator


class BlockStackLogger(Logger):
    # logger to save visualizations of input and output trajectories in block stacking environment

    @staticmethod
    def _init_env_from_id(id):
        # TODO: return different environment variants depending on id
        task_params =  AttrDict(
            max_tower_height=4
        )

        env_config = AttrDict(
            task_generator=FixedSizeSingleTowerBlockTaskGenerator,
            task_params=task_params,
            dimension=2,
            screen_width=128,
            screen_height=128
        )

        return BlockStackEnv(env_config)

    @staticmethod
    def _render_state(env, model_xml, obs, name=""):
        env.reset()

        unwrapped_obs = env._unflatten_block_obs(obs)

        sim_state = env.sim.get_state()

        sim_state.qpos[:len(sim_state.qpos)] = env.obs2qpos(obs)
        env.sim.set_state(sim_state)
        env.sim.forward()
        img = env.render()

        # round function
        rd = lambda x: np.round(x, 2)

        # add caption to the image
        info = {
            "Robot Pos": rd(unwrapped_obs["gripper_pos"]),
            "Robot Ang": rd(unwrapped_obs["gripper_angle"]),
            "Gripper Finger Pos": rd(unwrapped_obs["gripper_finger_pos"]),
        }

        for i in range(unwrapped_obs["block_pos"].shape[0]):
            info.update({
                "Block {}:".format(i): rd(unwrapped_obs["block_pos"][i])
            })

        img = add_caption_to_img(img, info, name, flip_rgb=True)

        return img


class SkillSpaceBlockStackLogger(BlockStackLogger, SkillSpaceLogger):
    pass
