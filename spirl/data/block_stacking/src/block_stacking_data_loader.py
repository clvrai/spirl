import os

from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset, GlobalSplitStateSequenceDataset


class BlockStackSequenceDataset(GlobalSplitVideoDataset):
    """Adds info about env idx from file path."""
    def _get_aux_info(self, data, path):
        # extract env name from file path
        # TODO: design an env id system for block stacking envs
        return AttrDict(env_id=0)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        for key in data.keys():
            if key.endswith('states') and data[key].shape[-1] == 40:
                # remove quatenion dimensions
                data[key] = data[key][:, :20]
            elif key.endswith('states') and data[key].shape[-1] == 43:
                data[key] = data[key][:, :23]
            if key.endswith('actions') and data[key].shape[-1] == 4:
                # remove rotation dimension
                data[key] = data[key][:, [0, 1, 3]]
        return data


class BlockStackStateSequenceDataset(BlockStackSequenceDataset, GlobalSplitStateSequenceDataset):
    pass
