import os

from spirl.utils.general_utils import AttrDict

from spirl.data.block_stacking.src.demo_gen.block_stacking_demo_agent import BlockStackingDemoAgent
from spirl.data.block_stacking.src.block_stacking_env import BlockStackEnv
from spirl.data.block_stacking.src.block_task_generator import FixedSizeSingleTowerBlockTaskGenerator


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used for generating block stacking dataset'
SEED = 31

configuration = {
    'seed': SEED,
    'agent': BlockStackingDemoAgent,
    'environment': BlockStackEnv,
    'max_rollout_len': 250,
}
configuration = AttrDict(configuration)

# Task
task_params = AttrDict(
    max_tower_height=4,
    seed=SEED,
)

# Agent
agent_config = AttrDict(

)

# Dataset - Random data
data_config = AttrDict(

)

# Environment
env_config = AttrDict(
    task_generator=FixedSizeSingleTowerBlockTaskGenerator,
    task_params=task_params,
    dimension=2,
    n_steps=2,
    screen_width=32,
    screen_height=32,
    rand_task=True,
    rand_init_pos=True,
    camera_name='agentview',
)

