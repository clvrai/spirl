from collections import deque
import numpy as np

from spirl.utils.general_utils import ParamDict


class BlockTaskGenerator:
    def __init__(self, hp, n_blocks):
        self._hp = self._default_hparams().overwrite(hp)
        self._n_blocks = n_blocks
        self._rng = np.random.default_rng(seed=self._hp.seed)

    def _default_hparams(self):
        default_dict = ParamDict({
            'seed': None,
        })
        return default_dict

    def sample(self):
        """Generates task definition in the form of list of transition tuples.
           Each tuple contains (bottom_block, top_block)."""
        raise NotImplementedError

    def _sample_tower(self, size, blocks):
        """Samples single tower of specified size, pops blocks from queue of idxs."""
        block = blocks.popleft()
        tasks = []
        for _ in range(size):
            next_block = blocks.popleft()
            tasks.append((block, next_block))
            block = next_block
        return tasks

    def _init_block_queue(self):
        return deque(self._rng.permutation(self._n_blocks))


class SingleTowerBlockTaskGenerator(BlockTaskGenerator):
    """Samples single tower with a limit on the number of stacked blocks."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'max_tower_height': 4,     # number of blocks in env
        })
        return super()._default_hparams().overwrite(default_dict)

    def sample(self):
        block_queue = self._init_block_queue()
        size = self._sample_size()
        return self._sample_tower(size, block_queue)

    def _sample_size(self):
        return self._rng.integers(1, self._hp.max_tower_height + 1)  # assure at least one stack is performed


class FixedSizeSingleTowerBlockTaskGenerator(SingleTowerBlockTaskGenerator):
    """Samples single tower, always samples maximum height."""
    def _sample_size(self):
        return self._hp.max_tower_height


class MultiTowerBlockTaskGenerator(BlockTaskGenerator):
    """Samples multiple tower with a limit on the number of stacked blocks."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'max_tower_height': 4,     # maximum height of target tower(s)
        })
        return super()._default_hparams().overwrite(default_dict)

    def sample(self):
        block_queue = self._init_block_queue()
        task = []
        while len(block_queue) > 1:
            size = self._sample_size(max_height=min(len(block_queue)-1, self._hp.max_tower_height))
            task += self._sample_tower(size, block_queue)
        return task

    def _sample_size(self, max_height):
        return self._rng.integers(1, max_height + 1)  # assure at least one stack is performed


if __name__ == '__main__':
    task_gen = MultiTowerBlockTaskGenerator(hp={}, n_blocks=10)
    for _ in range(10):
        print(task_gen.sample())
