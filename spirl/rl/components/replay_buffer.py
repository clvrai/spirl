import numpy as np
import gzip
import pickle
import os
import copy

from spirl.utils.general_utils import AttrDict, RecursiveAverageMeter, ParamDict


class ReplayBuffer:
    """Stores arbitrary rollout outputs that are provided by AttrDicts."""
    def __init__(self, hp):
        # TODO upgrade to more efficient (vectorized) implementation of rollout storage
        self._hp = self._default_hparams().overwrite(hp)
        self._max_capacity = self._hp.capacity
        self._replay_buffer = None
        self._idx = None
        self._size = None       # indicates whether all slots in replay buffer were filled at least once

    def _default_hparams(self):
        default_dict = ParamDict({
            'capacity': 1e6,        # max number of experience samples
            'dump_replay': True,    # whether replay buffer gets dump upon checkpointing
        })
        return default_dict

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            self._replay_buffer[key][idxs] = np.stack(experience_batch[key])

        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def sample(self, n_samples, filter=None):
        """Samples n_samples from the rollout_storage. Potentially can filter which fields to return."""
        raise NotImplementedError("Needs to be implemented by child class!")

    def get(self):
        """Returns complete replay buffer."""
        return self._replay_buffer

    def reset(self):
        """Deletes all entries from replay buffer and reinitializes."""
        del self._replay_buffer
        self._replay_buffer, self._idx, self._size = None, None, None

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._replay_buffer = AttrDict()
        for key in example_batch:
            example_element = example_batch[key][0]
            self._replay_buffer[key] = np.empty([int(self._max_capacity)] + list(example_element.shape),
                                                   dtype=example_element.dtype)
        self._idx = 0
        self._size = 0

    def save(self, save_dir):
        """Stores compressed replay buffer to file."""
        if not self._hp.dump_replay: return
        os.makedirs(save_dir, exist_ok=True)
        with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'wb') as f:
            pickle.dump(self._replay_buffer, f)
        np.save(os.path.join(save_dir, "idx_size.npy"), np.array([self._idx, self.size]))

    def load(self, save_dir):
        """Loads replay buffer from compressed disk file."""
        assert self._replay_buffer is None      # cannot overwrite existing replay buffer when loading
        if not self._hp.dump_replay:
            return
        with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'rb') as f:
            self._replay_buffer = pickle.load(f)
        idx_size = np.load(os.path.join(save_dir, "idx_size.npy"))
        self._idx, self._size = int(idx_size[0]), int(idx_size[1])

    @staticmethod
    def _get_n_samples(batch):
        """Retrieves the number of samples in batch."""
        for key in batch:
            return len(batch[key])

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._max_capacity

    def __contains__(self, key):
        return key in self._replay_buffer


class UniformReplayBuffer(ReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""
    def sample(self, n_samples, filter=None):
        assert n_samples <= self.size      # need enough samples in replay buffer
        assert isinstance(self.size, int)   # need integer-valued size
        idxs = np.random.choice(np.arange(self.size), size=n_samples)

        sampled_transitions = AttrDict()
        for key in self._replay_buffer:
            if filter is None or key in filter:
                sampled_transitions[key] = self._replay_buffer[key][idxs]
        return sampled_transitions


class FilteredReplayBuffer(ReplayBuffer):
    """Has option to *not* store certain attributes in replay (eg to save memory by not storing images."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'filter_keys': [],        # list of keys who's values should not get stored in replay
        })
        return default_dict

    def append(self, experience_batch):
        return super().append(AttrDict({k: v for (k,v) in experience_batch.items() if k not in self._hp.filter_keys}))


class FilteredUniormReplayBuffer(FilteredReplayBuffer, UniformReplayBuffer):
    def sample(self, n_samples, filter=None):
        return UniformReplayBuffer.sample(self, n_samples, filter)


class SplitObsReplayBuffer(ReplayBuffer):
    """Splits off unused part of observation before storing (eg to save memory by not storing images)."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'unused_obs_size': None,    # dimensionality of split off observation part
            'discard_part': 'back',     # which part of observation to discard ['front', 'back']
        })
        return super()._default_hparams().overwrite(default_dict)

    def append(self, experience_batch):
        filtered_experience_batch = copy.deepcopy(experience_batch)
        if self._hp.discard_part == 'front':
            filtered_experience_batch.observation = [o[self._hp.unused_obs_size:] for o in filtered_experience_batch.observation]
            filtered_experience_batch.observation_next = [o[self._hp.unused_obs_size:] for o in filtered_experience_batch.observation_next]
        elif self._hp.discard_part == 'back':
            filtered_experience_batch.observation = [o[:-self._hp.unused_obs_size] for o in filtered_experience_batch.observation]
            filtered_experience_batch.observation_next = [o[:-self._hp.unused_obs_size] for o in filtered_experience_batch.observation_next]
        else:
            raise ValueError("Cannot parse discard_part parameter {}!".format(self._hp.discard_part))
        return super().append(filtered_experience_batch)


class SplitObsUniformReplayBuffer(SplitObsReplayBuffer, UniformReplayBuffer):
    def sample(self, n_samples, filter=None):
        return UniformReplayBuffer.sample(self, n_samples, filter)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stats.update(AttrDict(
                avg_reward=np.stack(rollout.reward).sum()
            ))
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]






