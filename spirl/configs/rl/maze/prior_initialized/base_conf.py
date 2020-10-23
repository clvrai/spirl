from spirl.configs.rl.maze.base_conf import *
from spirl.rl.components.sampler import ACMultiImageAugmentedSampler
from spirl.rl.policies.mlp_policies import ConvPolicy
from spirl.rl.components.critic import SplitObsMLPCritic
from spirl.models.bc_mdl import ImageBCMdl


# update sampler
configuration['sampler'] = ACMultiImageAugmentedSampler
sampler_config = AttrDict(
    n_frames=2,
)
env_config.screen_width = data_spec.res
env_config.screen_height = data_spec.res

# update policy to conv policy
agent_config.policy = ConvPolicy
policy_params.update(AttrDict(
    input_nc=3 * sampler_config.n_frames,
    prior_model=ImageBCMdl,
    prior_model_params=AttrDict(state_dim=data_spec.state_dim,
                                action_dim=data_spec.n_actions,
                                input_res=data_spec.res,
                                n_input_frames=sampler_config.n_frames,
                        ),
    prior_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                        "skill_prior_learning/maze/flat"),
))

# update critic+policy to handle multi-frame combined observation
agent_config.critic = SplitObsMLPCritic
agent_config.critic_params.unused_obs_size = 32**2*3 * sampler_config.n_frames

