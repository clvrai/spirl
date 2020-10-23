import numpy as np

from spirl.utils.general_utils import ParamDict


class Normalizer:
    """Normalizes quantities (zero-mean, unit-variance)."""
    MIN_STD = 1e-2      # minimum standard deviation

    def __init__(self, hp):
        self._hp = self._default_hparams().overwrite(hp)
        self._sum, self._square_sum = None, None
        self._count = 0
        self._mean, self._std = 0, 1.0

    def _default_hparams(self):
        default_dict = ParamDict({
            'clip_raw_obs': np.array(float("Inf")),        # symmetric value maximum for raw observation
            'clip_norm_obs': np.array(float("Inf")),       # symmetric value maximum for normalized observation
            'update_horizon': 1e7,     # number of values for which statistics get updated
        })
        return default_dict

    def __call__(self, vals):
        """Performs normalization."""
        vals = self._clip(vals, range=self._hp.clip_raw_obs)
        return self._clip((vals - self._mean) / self._std, range=self._hp.clip_norm_obs)

    def update(self, vals):
        """Add new values to internal value, update statistics."""
        if self._count >= self._hp.update_horizon: return

        if isinstance(vals, list):
            vals = np.stack(vals)

        # clip + update summed vals
        vals = self._clip(vals, range=self._hp.clip_raw_obs)
        sum_val, square_sum_val = vals.sum(axis=0), (vals**2).sum(axis=0)
        if self._sum is None:
            self._sum, self._square_sum = sum_val, square_sum_val
        else:
            self._sum += sum_val; self._square_sum += square_sum_val
        self._count += vals.shape[0]

        # update statistics
        self._mean = self._sum / self._count
        self._std = np.sqrt(np.maximum(self.MIN_STD**2 * np.ones_like(self._sum),
                                       self._square_sum / self._count - (self._sum / self._count)**2))

    def reset(self):
        self._sum, self._square_sum = None, None
        self._count = 0
        self._mean, self._std = 0, 0.1

    @staticmethod
    def _clip(val, range):
        return np.clip(val, -range, range)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalizer2(Normalizer):
    def __init__(self, hp, shape=()):
        super().__init__(hp)
        self._running_mean_std = RunningMeanStd(shape=shape)

    def reset(self, shape=()):
        self._running_mean_std = RunningMeanStd(shape=shape)

    def update(self, vals):
        if isinstance(vals, list):
            vals = np.stack(vals)
        self._running_mean_std.update(vals)

    def __call__(self, vals):
        """Performs normalization."""
        vals = self._clip(vals, range=self._hp.clip_raw_obs)
        return self._clip((vals - self._running_mean_std.mean) / np.sqrt(self._running_mean_std.var),
                          range=self._hp.clip_norm_obs)

    @property
    def mean(self):
        return self._running_mean_std.mean

    @property
    def std(self):
        return np.sqrt(self._running_mean_std.var)


class DummyNormalizer(Normalizer):
    def __call__(self, vals):
        return vals

    def update(self, vals):
        pass


if __name__ == "__main__":
    norm1 = Normalizer({})
    norm2 = Normalizer2({})

    for _ in range(100):
        data = np.random.rand(16, 20, 4)
        norm1.update(data)
        norm2.update(data)

    print(np.abs(norm1._mean - norm2._running_mean_std.mean).max())
    print("##############")
    print(np.abs(norm1._std - np.sqrt(norm2._running_mean_std.var)).max())

