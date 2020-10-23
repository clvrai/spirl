from spirl.configs.rl.maze.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import ACPriorInitializedPolicy
from spirl.data.maze.src.maze_agents import MazeSACAgent

agent_config.policy = ACPriorInitializedPolicy
configuration.agent = MazeSACAgent
