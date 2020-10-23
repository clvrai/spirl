import os

from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.data.block_stacking.src.block_stacking_logger import SkillSpaceBlockStackLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.block_stacking import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageSkillPriorMdl,
    'logger': SkillSpaceBlockStackLogger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'block_stacking'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames
