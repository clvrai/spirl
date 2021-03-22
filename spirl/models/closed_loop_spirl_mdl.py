import torch

from spirl.utils.general_utils import batch_apply
from spirl.utils.pytorch_utils import get_constant_parameter
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM
from spirl.modules.variational_inference import MultivariateGaussian


class ClSPiRLMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self._hp.state_dim + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0, learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        decode_inputs = torch.cat((inputs.states[:, :steps],
                                   z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.state_dim
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, inputs.states[:, :-1]), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])
