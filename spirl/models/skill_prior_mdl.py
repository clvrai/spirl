from contextlib import contextmanager
import itertools
import copy
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from spirl.components.base_model import BaseModel
from spirl.components.logger import Logger
from spirl.components.checkpointer import load_by_key, freeze_modules
from spirl.modules.losses import KLDivLoss, NLL
from spirl.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from spirl.modules.recurrent_modules import RecurrentPredictor
from spirl.utils.general_utils import AttrDict, ParamDict, split_along_axis, get_clipped_optimizer
from spirl.utils.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor, \
                                        TensorModule, RAdam
from spirl.utils.vis_utils import fig2img
from spirl.modules.variational_inference import ProbabilisticModel, Gaussian, MultivariateGaussian, get_fixed_prior, \
                                                mc_kl_divergence
from spirl.modules.layers import LayerBuilderParams
from spirl.modules.mdn import MDN, GMM
from spirl.modules.flow_models import ConditionedFlowModel


class SkillPriorMdl(BaseModel, ProbabilisticModel):
    """Skill embedding + prior model for SPIRL algorithm."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device

        self.build_network()

        # optionally: optimize beta with dual gradient descent
        if self._hp.target_kl is not None:
            self._log_beta = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
            self._beta_opt = self._get_beta_opt()

        self.load_weights_and_freeze()

    @contextmanager
    def val_mode(self):
        self.switch_to_prior()
        yield
        self.switch_to_inference()

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict({
            'use_convs': False,
            'device': None,
            'n_rollout_steps': 10,        # number of decoding steps
            'cond_decode': False,         # if True, conditions decoder on prior inputs
        })

        # Network size
        default_dict.update({
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_enc': 32,               # number of dimensions in encoder-latent space
            'nz_vae': 10,               # number of dimensions in vae-latent space
            'nz_mid': 32,               # number of dimensions for internal feature spaces
            'nz_mid_lstm': 128,         # size of middle LSTM layers
            'n_lstm_layers': 1,         # number of LSTM layers
            'n_processing_layers': 3,   # number of layers in MLPs
        })

        # Learned prior
        default_dict.update({
            'n_prior_nets': 1,              # number of prior networks in ensemble
            'num_prior_net_layers': 6,      # number of layers of the learned prior MLP
            'nz_mid_prior': 128,            # dimensionality of internal feature spaces for prior net
            'nll_prior_train': True,        # if True, trains learned prior by maximizing NLL
            'learned_prior_type': 'gauss',  # distribution type for learned prior, ['gauss', 'gmm', 'flow']
            'n_gmm_prior_components': 5,    # number of Gaussian components for GMM learned prior
        })

        # Loss weights
        default_dict.update({
            'reconstruction_mse_weight': 1.,    # weight of MSE reconstruction loss
            'kl_div_weight': 1.,                # weight of KL divergence loss
            'target_kl': None,                  # if not None, adds automatic beta-tuning to reach target KL divergence
        })

        # loading pre-trained components
        default_dict.update({
            'embedding_checkpoint': None,   # optional, if provided loads weights for encoder, decoder and freezes it
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        """Defines the network architecture (encoder aka inference net, decoder, prior)."""
        assert not self._hp.use_convs   # currently only supports non-image inputs
        self.q = self._build_inference_net()
        self.decoder = RecurrentPredictor(self._hp,
                                          input_size=self._hp.action_dim+self._hp.nz_vae,
                                          output_size=self._hp.action_dim)
        self.decoder_input_initalizer = self._build_decoder_initializer(size=self._hp.action_dim)
        self.decoder_hidden_initalizer = self._build_decoder_initializer(size=self.decoder.cell.get_state_size())

        self.p = self._build_prior_ensemble()

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions    # for seamless evaluation

        # run inference
        output.q = self._run_inference(inputs)

        # compute (fixed) prior
        output.p = get_fixed_prior(output.q)

        # infer learned skill prior
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        if use_learned_prior:
            output.p = output.q_hat     # use output of learned skill prior for sampling

        # sample latent variable
        output.z = output.p.sample() if self._sample_prior else output.q.sample()
        output.z_q = output.z.clone() if not self._sample_prior else output.q.sample()   # for loss computation

        # decode
        assert self._regression_targets(inputs).shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        return output

    def loss(self, model_output, inputs):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight) \
            (Gaussian(model_output.reconstruction, torch.zeros_like(model_output.reconstruction)),
             self._regression_targets(inputs))

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)

        # learned skill prior net loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        self._logger.log_scalar(self.beta, "beta", step, phase)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(model_output, inputs, losses, step, phase, logger, **logging_kwargs)

    def decode(self, z, cond_inputs, steps, inputs=None):
        """Runs forward pass of decoder given skill embedding.
        :arg z: skill embedding
        :arg cond_inputs: info that decoder is conditioned on
        :arg steps: number of steps decoder is rolled out
        """
        lstm_init_input = self.decoder_input_initalizer(cond_inputs)
        lstm_init_hidden = self.decoder_hidden_initalizer(cond_inputs)
        return self.decoder(lstm_initial_inputs=AttrDict(x_t=lstm_init_input),
                            lstm_static_inputs=AttrDict(z=z),
                            steps=steps,
                            lstm_hidden_init=lstm_init_hidden).pred

    def run(self, inputs, use_learned_prior=True):
        """Policy interface for model. Runs decoder if action plan is empty, otherwise returns next action from action plan.
        :arg inputs: dict with 'states', 'actions', 'images' keys from environment
        :arg use_learned_prior: if True, uses learned prior otherwise samples latent from uniform prior
        """
        if not self._action_plan:
            inputs = map2torch(inputs, device=self.device)

            # sample latent variable from prior
            z = self.compute_learned_prior(self._learned_prior_input(inputs), first_only=True).sample() \
                if use_learned_prior else Gaussian(torch.zeros((1, self._hp.nz_vae*2), device=self.device)).sample()

            # decode into action plan
            z = z.repeat(self._hp.batch_size, 1)  # this is a HACK flat LSTM decoder can only take batch_size inputs
            input_obs = self._learned_prior_input(inputs).repeat(self._hp.batch_size, 1)
            actions = self.decode(z, cond_inputs=input_obs, steps=self._hp.n_rollout_steps)[0]
            self._action_plan = deque(split_along_axis(map2np(actions), axis=0))

        return AttrDict(action=self._action_plan.popleft()[None])

    def reset(self):
        """Resets action plan (should be called at beginning of episode when used in RL loop)."""
        self._action_plan = deque()        # stores action plan of LL policy when model is used as policy

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.decoder_input_initalizer, self.decoder_hidden_initalizer, self.q])

    def _build_inference_net(self):
        # inference gets conditioned on state if decoding is also conditioned on state
        input_size = self._hp.action_dim + self.prior_input_size if self._hp.cond_decode else self._hp.action_dim
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _build_decoder_initializer(self, size):
        if self._hp.cond_decode:
            # roughly match parameter count of the learned prior
            return Predictor(self._hp, input_size=self.prior_input_size, output_size=size,
                             num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)
        else:
            class FixedTrainableInitializer(nn.Module):
                def __init__(self, hp):
                    super().__init__()
                    self._hp = hp
                    self.val = nn.Parameter(torch.zeros((1, size), requires_grad=True, device=self._hp.device))

                def forward(self, state):
                    return self.val.repeat(find_tensor(state).shape[0], 1)
            return FixedTrainableInitializer(self._hp)

    def _build_prior_ensemble(self):
        return nn.ModuleList([self._build_prior_net() for _ in range(self._hp.n_prior_nets)])

    def _build_prior_net(self):
        """Supports building Gaussian, GMM and Flow prior networks. Default is Gaussian skill prior."""
        if self._hp.learned_prior_type == 'gmm':
            return torch.nn.Sequential(
                Predictor(self._hp, input_size=self.prior_input_size, output_size=self._hp.nz_mid,
                          num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior),
                MDN(input_size=self._hp.nz_mid, output_size=self._hp.nz_vae,
                    num_gaussians=self._hp.n_gmm_prior_components)
            )
        elif self._hp.learned_prior_type == 'flow':
            return ConditionedFlowModel(self._hp, input_dim=self.prior_input_size, output_dim=self._hp.nz_vae,
                                        n_flow_layers=self._hp.num_prior_net_layers)
        else:
            return Predictor(self._hp, input_size=self.prior_input_size, output_size=self._hp.nz_vae * 2,
                             num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)

    def _run_inference(self, inputs):
        inf_input = inputs.actions
        if self._hp.cond_decode:
            inf_input = torch.cat((inf_input, self._learned_prior_input(inputs)[:, None]
                                        .repeat(1, inf_input.shape[1], 1)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def compute_learned_prior(self, inputs, first_only=False):
        """Splits batch into separate batches for prior ensemble, optionally runs first or avg prior on whole batch.
           (first_only, avg == True is only used for RL)."""
        if first_only:
            return self._compute_learned_prior(self.p[0], inputs)

        assert inputs.shape[0] % self._hp.n_prior_nets == 0
        per_prior_inputs = torch.chunk(inputs, self._hp.n_prior_nets)
        prior_results = [self._compute_learned_prior(prior, input_batch)
                         for prior, input_batch in zip(self.p, per_prior_inputs)]

        return type(prior_results[0]).cat(*prior_results, dim=0)

    def _compute_learned_prior(self, prior_mdl, inputs):
        if self._hp.learned_prior_type == 'gmm':
            return GMM(*prior_mdl(inputs))
        elif self._hp.learned_prior_type == 'flow':
            return prior_mdl(inputs)
        else:
            return MultivariateGaussian(prior_mdl(inputs))

    def _compute_learned_prior_loss(self, model_output):
        if self._hp.nll_prior_train:
            loss = NLL(breakdown=0)(model_output.q_hat, model_output.z_q.detach())
        else:
            loss = KLDivLoss(breakdown=0)(model_output.q.detach(), model_output.q_hat)
        # aggregate loss breakdown for each of the priors in the ensemble
        loss.breakdown = torch.stack([chunk.mean() for chunk in torch.chunk(loss.breakdown, self._hp.n_prior_nets)])
        return loss

    def _get_beta_opt(self):
        return get_clipped_optimizer(filter(lambda p: p.requires_grad, self._log_beta.parameters()),
                                     lr=3e-4, optimizer_type=RAdam, betas=(0.9, 0.999), gradient_clip=None)

    def _update_beta(self, kl_div):
        """Updates beta with dual gradient descent."""
        assert self._hp.target_kl is not None
        beta_loss = self._log_beta().exp() * (self._hp.target_kl - kl_div).detach().mean()
        self._beta_opt.zero_grad()
        beta_loss.backward()
        self._beta_opt.step()

    def _learned_prior_input(self, inputs):
        return inputs.states[:, 0]

    def _regression_targets(self, inputs):
        return inputs.actions

    def evaluate_prior_divergence(self, state):
        """Evaluates prior divergence as mean pairwise KL divergence of ensemble of priors."""
        assert self._hp.n_prior_nets > 1        # need more than one prior in ensemble to evaluate divergence
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self._hp.device)
        if len(state.shape) == 1:
            state = state[None]
        state_batch = state.repeat(self._hp.n_prior_nets, 1) if len(state.shape) == 1 else \
            state.repeat(self._hp.n_prior_nets, 1, 1, 1)
        priors = self.compute_learned_prior(state_batch).chunk(self._hp.n_prior_nets)
        divergences = [mc_kl_divergence(*pair) for pair in itertools.permutations(priors, r=2)]
        return torch.mean(torch.stack(divergences), dim=0)

    @property
    def resolution(self):
        return 64       # return dummy resolution, images are not used by this model

    @property
    def latent_dim(self):
        return self._hp.nz_vae

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def prior_input_size(self):
        return self.state_dim

    @property
    def n_rollout_steps(self):
        return self._hp.n_rollout_steps

    @property
    def beta(self):
        return self._log_beta().exp()[0].detach() if self._hp.target_kl is not None else self._hp.kl_div_weight


class ImageSkillPriorMdl(SkillPriorMdl):
    """Implements learned skill prior with image input."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'prior_input_res': 32,      # input resolution of prior images
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed flat we are not reconstructing
            img_sz=self._hp.prior_input_res,  # image resolution
            input_nc=3*self._hp.n_input_frames,  # number of input feature maps
            ngf=self._hp.encoder_ngf,         # number of feature maps in shallowest level
            nz_enc=self.prior_input_size,     # size of image encoder output feature
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))

    def _build_prior_net(self):
        return nn.Sequential(
            ResizeSpatial(self._hp.prior_input_res),
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            super()._build_prior_net(),
        )

    def _build_inference_net(self):
        # inference gets conditioned on prior input if decoding is also conditioned on prior input
        if not self._hp.cond_decode:
            return super()._build_inference_net()
        self.cond_encoder = nn.Sequential(ResizeSpatial(self._hp.prior_input_res),      # encodes image inputs
                                          Encoder(self._updated_encoder_params()),
                                          RemoveSpatial(),)
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=self._hp.action_dim + self._hp.nz_enc, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _build_decoder_initializer(self, size):
        if not self._hp.cond_decode:
            return super()._build_decoder_initializer(size)
        return nn.Sequential(
            self.cond_encoder,      # encode image conditioning
            super()._build_decoder_initializer(size),
        )

    def _run_inference(self, inputs):
        if not self._hp.cond_decode:
            return super()._run_inference(inputs)
        enc_cond = self.cond_encoder(self._learned_prior_input(inputs))
        inf_input = torch.cat((inputs.actions, enc_cond[:, None].repeat(1, inputs.actions.shape[1], 1)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def _learned_prior_input(self, inputs):
        return inputs.images[:, :self._hp.n_input_frames]\
            .reshape(inputs.images.shape[0], -1, self.resolution, self.resolution)

    def _regression_targets(self, inputs):
        return inputs.actions[:, (self._hp.n_input_frames-1):]

    def unflatten_obs(self, raw_obs):
        """Utility to unflatten [obs, prior_obs] concatenated observation (for RL usage)."""
        assert len(raw_obs.shape) == 2 and raw_obs.shape[1] == self.state_dim \
               + self._hp.prior_input_res**2 * 3 * self._hp.n_input_frames
        return AttrDict(
            obs=raw_obs[:, :self.state_dim],
            prior_obs=raw_obs[:, self.state_dim:].reshape(raw_obs.shape[0], 3*self._hp.n_input_frames,
                                                          self._hp.prior_input_res, self._hp.prior_input_res)
        )

    @property
    def prior_input_size(self):
        return self._hp.nz_mid

    @property
    def resolution(self):
      return self._hp.prior_input_res


class SkillSpaceLogger(Logger):
    """
    Logger for Skill Space model. No extra methods needed to implement by
    environment-specific logger implementation.
    """
    def visualize(self, model_output, inputs, losses, step, phase, logger):
        self._plot_latents(model_output, logger, step, phase)

    def _plot_latents(self, model_output, logger, step, phase):
        """Visualizes 2D Gaussian latents if available."""
        if model_output.p.shape[1] == 2:   # only supports 2D gaussian latents
            graphs = []
            for i in range(self._n_logged_samples):
                fig = plt.figure()
                ax = plt.subplot(111)
                plt.xlim(-2, 2); plt.ylim(-2, 2)

                # draw prior
                self._draw_gaussian(ax, model_output.p[i].tensor(), color='black')

                # draw posterior
                self._draw_gaussian(ax, model_output.q[i].tensor(), color='red')

                # draw estimated posterior
                if 'q_hat' in model_output:
                    self._draw_learned_prior(ax, model_output.q_hat[i], color='green')

                graphs.append(fig2img(fig))
                plt.close()
            logger.log_images(np.stack(graphs), "latent_dists", step, phase)

    @staticmethod
    def _draw_gaussian(ax, gauss_tensor, color, weight=None):
        px, py, p_logsig_x, p_logsig_y = split_along_axis(ten2ar(gauss_tensor), axis=0)

        def logsig2std(logsig):
            return np.exp(logsig)

        ell = Ellipse(xy=(px, py),
                      width=2*logsig2std(p_logsig_x), height=2*logsig2std(p_logsig_y),
                      angle=0, color=color)     # this assumes diagonal gaussian
        if weight is not None:
            ell.set_alpha(weight)
        else:
            ell.set_facecolor('none')
        ax.add_artist(ell)

    def _draw_learned_prior(self, ax, prior, color):
        if isinstance(prior, GMM):
            [self._draw_gaussian(ax, component.tensor(), color, ten2ar(weight)) for weight, component in prior]
        else:
            self._draw_gaussian(ax, prior.tensor(), color)
