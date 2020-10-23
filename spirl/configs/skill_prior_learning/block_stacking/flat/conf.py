import os

from spirl.models.bc_mdl import ImageBCMdl
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.block_stacking import data_spec
from spirl.components.evaluator import DummyEvaluator
from spirl.components.logger import Logger


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageBCMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'block_stacking'),
    'epoch_cycles_train': 4,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    n_input_frames=2,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1 + 1 + (model_config.n_input_frames - 1)
