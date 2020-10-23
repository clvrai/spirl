from spirl.configs.rl.block_stacking.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

# update agent
agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = ActionPriorSACAgent