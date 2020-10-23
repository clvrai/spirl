from spirl.configs.hrl.kitchen.base_conf import *
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent


# update policy to use prior model for computing divergence
hl_policy_params.update(AttrDict(
    prior_model=ll_agent_config.model,
    prior_model_params=ll_agent_config.model_params,
    prior_model_checkpoint=ll_agent_config.model_checkpoint,
))
hl_agent_config.policy = LearnedPriorAugmentedPIPolicy

# update agent, set target divergence
agent_config.hl_agent = ActionPriorSACAgent
agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=5.),
))
