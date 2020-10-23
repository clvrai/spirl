from spirl.configs.rl.block_stacking.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import ACPriorInitializedPolicy

# update agent
agent_config.policy = ACPriorInitializedPolicy
configuration.agent = SACAgent
