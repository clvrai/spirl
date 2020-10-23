import torch

from spirl.modules.variational_inference import MultivariateGaussian
from spirl.rl.components.policy import Policy
from spirl.utils.general_utils import ParamDict


class UniformGaussianPolicy(Policy):
    """Samples actions from a uniform Gaussian."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        super().__init__()

    def _default_hparams(self):
        default_dict = ParamDict({
            'scale': 1,                         # scale of Uniform Gaussian
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return torch.nn.Module()        # dummy module

    def _compute_action_dist(self, obs):
        batch_size = obs.shape[0]
        return MultivariateGaussian(mu=torch.zeros((batch_size, self._hp.action_dim), device=obs.device),
                                    log_sigma=torch.log(self._hp.scale *
                                                        torch.ones((batch_size, self._hp.action_dim), device=obs.device)))
