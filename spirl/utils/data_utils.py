import numpy as np


class DataSubsampler:
    def __init__(self, aggregator):
        self._aggregator = aggregator

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class FixedFreqSubsampler(DataSubsampler):
    """Subsamples input array's first dimension by skipping given number of frames."""
    def __init__(self, n_skip, aggregator=None):
        super().__init__(aggregator)
        self._n_skip = n_skip

    def __call__(self, val, idxs=None, aggregate=False):
        """Subsamples with idxs if given, aggregates with aggregator if aggregate=True."""
        if self._n_skip == 0:
            return val, None

        if idxs is None:
            seq_len = val.shape[0]
            idxs = np.arange(0, seq_len - 1, self._n_skip + 1)

        if aggregate:
            assert self._aggregator is not None     # no aggregator given!
            return self._aggregator(val, idxs), idxs
        else:
            return val[idxs], idxs


class Aggregator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class SumAggregator(Aggregator):
    def __call__(self, val, idxs):
        return np.add.reduceat(val, idxs, axis=0)

