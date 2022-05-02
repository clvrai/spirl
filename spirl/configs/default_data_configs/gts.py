from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitActionSequenceDataset


from spirl.utils.gts_utils import state_dim

data_spec = AttrDict(
    dataset_class=GlobalSplitActionSequenceDataset,
    n_actions=2,
    state_dim=state_dim,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    res=32,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 300
