import torch
import torch.nn as nn

from spirl.utils.general_utils import batch_apply, ParamDict
from spirl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from spirl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian


class ClSPiRLMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0, learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    @property
    def enc_size(self):
        return self._hp.state_dim


class ImageClSPiRLMdl(ClSPiRLMdl, ImageSkillPriorMdl):
    """SPiRL model with closed-loop decoder that operates on image observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'prior_input_res': 32,      # input resolution of prior images
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _build_prior_net(self):
        return ImageSkillPriorMdl._build_prior_net(self)

    def _build_inference_net(self):
        self.img_encoder = nn.Sequential(ResizeSpatial(self._hp.prior_input_res),  # encodes image inputs
                                         Encoder(self._updated_encoder_params()),
                                         RemoveSpatial(),)
        return ClSPiRLMdl._build_inference_net(self)

    def _get_seq_enc(self, inputs):
        # stack input image sequence
        stacked_imgs = torch.cat([inputs.images[:, t:t+inputs.actions.shape[1]]
                                  for t in range(self._hp.n_input_frames)], dim=2)
        # encode stacked seq
        return batch_apply(stacked_imgs, self.img_encoder)

    def _learned_prior_input(self, inputs):
        return ImageSkillPriorMdl._learned_prior_input(self, inputs)

    def _regression_targets(self, inputs):
        return ImageSkillPriorMdl._regression_targets(self, inputs)

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return self.img_encoder(obs)

    @property
    def enc_size(self):
        return self._hp.nz_enc

    @property
    def prior_input_size(self):
        return self.enc_size


