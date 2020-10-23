from spirl.configs.rl.kitchen.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = LearnedPriorAugmentedPIPolicy
configuration.agent = ActionPriorSACAgent
