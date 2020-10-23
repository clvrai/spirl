import torch.nn as nn
import copy

from spirl.modules.layers import LayerBuilderParams
from spirl.modules.mdn import MDN, GMM
from spirl.modules.subnetworks import Predictor, HybridConvMLPEncoder, Encoder
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.rl.components.policy import Policy
from spirl.utils.general_utils import ParamDict, AttrDict
from spirl.utils.pytorch_utils import RemoveSpatial


class MLPPolicy(Policy):
    """MLP-based Gaussian policy."""
    def __init__(self, config):
        # TODO automate the setup by getting params from the environment
        self._hp = self._default_hparams().overwrite(config)
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization=self._hp.normalization)
        super().__init__()

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_dim': 32,                  # dimensionality of the observation input
            'n_layers': 3,                    # number of policy network layers
            'nz_mid': 64,                     # size of the intermediate network layers
            'normalization': 'none',          # normalization used in policy network ['none', 'batch']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return Predictor(self._hp,
                         input_size=self._hp.input_dim,
                         output_size=self.mlp_output_size,
                         mid_size=self._hp.nz_mid,
                         num_layers=self._hp.n_layers,
                         final_activation=None,
                         spatial=False)

    def _compute_action_dist(self, obs):
        return MultivariateGaussian(self.net(obs))

    @property
    def mlp_output_size(self):
        return 2 * self._hp.action_dim      # scale and variance of Gaussian output


class MDNPolicy(MLPPolicy):
    """GMM Policy (based on mixture-density network)."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'num_gaussians': None,          # number of mixture components
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        assert self._hp.num_gaussians is not None   # need to specify number of mixture components for policy
        return nn.Sequential(
            super()._build_network(),
            MDN(self.mlp_output_size, self._hp.action_dim, self._hp.num_gaussians)
        )

    def _compute_action_dist(self, obs):
        return GMM(self.net(obs))

    @property
    def mlp_output_size(self):
        """Mean, variance and weight for each Gaussian."""
        return self._hp.nz_mid


class SplitObsMLPPolicy(MLPPolicy):
    """Splits off unused part of observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'unused_obs_size': None,    # dimensionality of split off observation part
            'discard_part': 'back',     # which part of observation to discard ['front', 'back']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_action_dist(self, raw_obs):
        if self._hp.discard_part == 'front':
            return super()._compute_action_dist(raw_obs[:, self._hp.unused_obs_size:])
        elif self._hp.discard_part == 'back':
            return super()._compute_action_dist(raw_obs[:, :-self._hp.unused_obs_size])
        else:
            raise ValueError("Cannot parse discard_part parameter {}!".format(self._hp.discard_part))


class ConvPolicy(MLPPolicy):
    """Enodes input image with conv encoder, then uses MLP head to produce output action distribution."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return nn.Sequential(
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            Predictor(self._hp,
                      input_size=self._hp.nz_enc,
                      output_size=self.mlp_output_size,
                      mid_size=self._hp.nz_mid,
                      num_layers=self._hp.n_layers,
                      final_activation=None,
                      spatial=False),
        )

    def _compute_action_dist(self, obs):
        return super()._compute_action_dist(
            obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res))

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed flat we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))


class HybridConvMLPPolicy(ConvPolicy):
    """Policy that can incorporate image and scalar inputs by fusing conv and MLP encoder."""
    def _build_network(self):
        return HybridConvMLPEncoder(self._hp.overwrite(AttrDict(output_dim=self.mlp_output_size)))

    def _compute_action_dist(self, obs):
        """Splits concatenated input obs into image and vector observation and passes through network."""
        split_obs = AttrDict(
            vector=obs[:, :self._hp.input_dim],
            image=obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
        )
        return super()._compute_action_dist(split_obs)

