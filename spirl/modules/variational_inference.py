import torch
import math
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from spirl.utils.pytorch_utils import ten2ar
from spirl.utils.general_utils import batch_apply


class Gaussian:
    """ Represents a gaussian distribution """
    # TODO: implement a dict conversion function
    def __init__(self, mu, log_sigma=None):
        """
        
        :param mu:
        :param log_sigma: If none, mu is divided into two chunks, mu and log_sigma
        """
        if log_sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb; pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, -1)
            
        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) if isinstance(log_sigma, torch.Tensor) else \
                            np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None
        
    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
               / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussian distribution."""
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    @staticmethod
    def stack(*argv, dim):
        return Gaussian._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return Gaussian._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussian(mu, log_sigma)

    def average(self, dists):
        """Fits single Gaussian to a list of Gaussians."""
        mu_avg = torch.stack([d.mu for d in dists]).sum(0) / len(dists)
        sigma_avg = torch.stack([d.mu ** 2 + d.sigma ** 2 for d in dists]).sum(0) - mu_avg**2
        return type(self)(mu_avg, torch.log(sigma_avg))

    def chunk(self, *args, **kwargs):
        return [type(self)(chunk) for chunk in torch.chunk(self.tensor(), *args, **kwargs)]

    def view(self, shape):
        self.mu = self.mu.view(shape)
        self.log_sigma = self.log_sigma.view(shape)
        self._sigma = self.sigma.view(shape)
        return self

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self.log_sigma[item])
 
    def tensor(self):
        return torch.cat([self.mu, self.log_sigma], dim=-1)

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(self.mu.detach(), self.log_sigma.detach())

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return type(self)(ten2ar(self.mu), ten2ar(self.log_sigma))


class UnitGaussian(Gaussian):
    def __init__(self, size, device):
        mu = torch.zeros(size, device=device)
        log_sigma = torch.zeros(size, device=device)
        super().__init__(mu, log_sigma)


class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

    @staticmethod
    def stack(*argv, dim):
        return MultivariateGaussian(Gaussian.stack(*argv, dim=dim).tensor())

    @staticmethod
    def cat(*argv, dim):
        return MultivariateGaussian(Gaussian.cat(*argv, dim=dim).tensor())


class MultivariateDiagNormal(MultivariateNormal):
    def __init__(self, loc, scale, *args, **kwargs):
        cov = torch.diag_embed(scale.pow(2))
        super().__init__(loc, cov, *args, **kwargs)


class SequentialGaussian_SharedPQ:
    """ stacks two Gaussians """
    def __init__(self, g1, z1, g2):
        """

        """
        self.g1 = g1
        self.g2 = g2
        self.z1 = z1
        assert z1.shape == g1.shape
        self.shared_dims = None     # how many shape dimensions are shared
        self._update_shared_dims()

    def sample(self):
        """
        sample z2 and concatentate with z1
        :return:
        """
        return torch.cat([self.z1, self.g2.sample()], dim=1)

    def kl_divergence(self, other):
        return self.g1.kl_divergence(other.g1)

    @property
    def shape(self):
        self._update_shared_dims()
        return self.g1.shape[:self.shared_dims]

    @property
    def mu(self):
        return self.g1.mu

    @staticmethod
    def stack(*argv, dim):
        return SequentialGaussian_SharedPQ._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return SequentialGaussian_SharedPQ._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        def fn_apply(inputs):
            mu, log_sigma = [], []
            for g in inputs:
                mu.append(g.mu)
                log_sigma.append(g.log_sigma)
            mu = fcn(mu, dim)
            log_sigma = fcn(log_sigma, dim)
            return Gaussian(mu, log_sigma)

        g1_list = [a.g1 for a in argv]
        g2_list = [a.g2 for a in argv]
        z1_list = [a.z1 for a in argv]
        return SequentialGaussian_SharedPQ(fn_apply(g1_list), fcn(z1_list, dim=dim), fn_apply(g2_list))

    def view(self, shape):
        # assume that this shape does not include the last dimensions
        self._update_shared_dims()
        self.g1 = self.g1.view(shape + list(self.g1.shape[self.shared_dims:]))
        self.g2 = self.g2.view(shape + list(self.g2.shape[self.shared_dims:]))
        self.z1 = self.z1.view(shape + list(self.z1.shape[self.shared_dims:]))
        return self

    def __getitem__(self, item):
        return SequentialGaussian_SharedPQ(self.g1[item], self.z1[item], self.g2[item])

    def _update_shared_dims(self):
        shared_dims = 0
        for i, j in zip(self.g1.shape, self.g2.shape):
            if i != j: break
            shared_dims += 1
        assert shared_dims is not 0  # need at least one shared dim between the Gaussians
        self.shared_dims = shared_dims


class ProbabilisticModel:
    def __init__(self):
        self._sample_prior = False
        
    def switch_to_prior(self):
        self._sample_prior = True

    def switch_to_inference(self):
        self._sample_prior = False
    

def get_fixed_prior(tensor, bs=None, dim=None):
    if dim is not None:
        return Gaussian(tensor.new_zeros(bs, dim, 1, 1), tensor.new_zeros(bs, dim, 1, 1))
    else:
        return Gaussian(torch.zeros_like(tensor.mu), torch.zeros_like(tensor.log_sigma))
    

def stack(inp, dim):
    if isinstance(inp[0], Gaussian):
        return Gaussian.stack(*inp, dim=dim)
    else:
        return torch.stack(inp, dim)


def mc_kl_divergence(p, q, n_samples=1):
    """Computes monte-carlo estimate of KL divergence. n_samples: how many samples are used for the estimate."""
    samples = [p.sample() for _ in range(n_samples)]
    return torch.stack([p.log_prob(x) - q.log_prob(x) for x in samples], dim=1).mean(dim=1)


if __name__ == "__main__":
    d = [Gaussian(torch.tensor([1., 1]), torch.zeros(2)), Gaussian(torch.tensor([-1., -1]), torch.zeros(2))]
    d_avg = Gaussian.average(d)
    print(d_avg.mu, d_avg.sigma)

