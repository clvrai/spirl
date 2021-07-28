import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.critic import MLPCritic
from spirl.rl.envs.office import OfficeEnv
from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from spirl.configs.default_data_configs.office import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the office env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': OfficeEnv,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 350,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 2e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=10, #256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy Params
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    nz_vae=10,
    nz_mid=128,
    n_processing_layers=5,
    kl_div_weight=5e-4,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/office/hierarchical_cl"),
    initial_log_sigma=-50,
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                           # LL critic is not used since we are not finetuning LL
    critic_params=ll_critic_params,
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=LearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
    td_schedule_params=AttrDict(p=1.),
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=False,
    update_hl=True,
    update_ll=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    name='Widow250OfficeFixed-v0',
)

