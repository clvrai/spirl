import torch
import torch.nn.functional as F


def safe_entropy(dist, dim=None, eps=1e-12):
    """Computes entropy even if some entries are 0."""
    return -torch.sum(dist * safe_log_prob(dist, eps), dim=dim)


def safe_log_prob(tensor, eps=1e-12):
    """Safe log of probability values (must be between 0 and 1)"""
    return torch.log(torch.clamp(tensor, eps, 1 - eps))


def normalize(tensor, dim=1, eps=1e-7):
    norm = torch.clamp(tensor.sum(dim, keepdim=True), eps)
    return tensor / norm


def gumbel_sample(shape, eps=1e-8):
    """Sample Gumbel noise."""
    uniform = torch.rand(shape).float()
    return -torch.log(eps - torch.log(uniform + eps))


def gumbel_softmax_sample(logits, temp=1.):
    """Sample from the Gumbel softmax / concrete distribution."""
    gumbel_noise = gumbel_sample(logits.size()).to(logits.device)
    return F.softmax((logits + gumbel_noise) / temp, dim=1)


def log_cumsum(probs, dim=1, eps=1e-8):
    """Calculate log of inclusive cumsum."""
    return torch.log(torch.cumsum(probs, dim=dim) + eps)


def poisson_categorical_log_prior(length, rate, device):
    """Categorical prior populated with log probabilities of Poisson dist.
       From: https://github.com/tkipf/compile/blob/b88b17411c37e1ed95459a0a779d71d5acef9e3f/utils.py#L58"""
    rate = torch.tensor(rate, dtype=torch.float32, device=device)
    values = torch.arange(1, length + 1, dtype=torch.float32, device=device).unsqueeze(0)
    log_prob_unnormalized = torch.log(rate) * values - rate - (values + 1).lgamma()
    # TODO(tkipf): Length-sensitive normalization.
    return F.log_softmax(log_prob_unnormalized, dim=1)  # Normalize.


def kl_categorical(preds, log_prior, eps=1e-8):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum(1)


class Dirac:
    """Dummy Dirac distribution."""
    def __init__(self, val):
        self._val = val

    def sample(self):
        return self._val

    def rsample(self):
        return self._val

    def log_prob(self, val):
        return torch.tensor(int(val == self._val), dtype=torch.float32, device=self._val.device)

    @property
    def logits(self):
        """This is more of a dummy return."""
        return self._val
