from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset


data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=8,
    state_dim=97,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    env_name="Widow250OfficeFixed-v0",
    res=64,
    crop_rand_subseq=True,
)

data_spec.max_seq_len = 350
