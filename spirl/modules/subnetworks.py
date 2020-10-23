import copy
import math
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spirl.modules.layers import BaseProcessingNet, ConvBlockEnc, \
    ConvBlockDec, init_weights_xavier, get_num_conv_layers, ConvBlockFirstDec, ConvBlock, LayerBuilderParams
from spirl.modules.recurrent_modules import BaseProcessingLSTM, \
    BidirectionalLSTM
from spirl.modules.variational_inference import Gaussian
from spirl.modules.variational_inference import SequentialGaussian_SharedPQ
from spirl.modules.variational_inference import UnitGaussian
from spirl.utils.general_utils import SkipInputSequential, GetIntermediatesSequential, \
    remove_spatial, batchwise_index, batch_apply, map_recursive, apply_linear, ParamDict
from spirl.utils.pytorch_utils import like, AttrDictPredictor, batchwise_assign, make_one_hot, mask_out
from spirl.utils.general_utils import broadcast_final, AttrDict
from torch import Tensor


class ParamLayer(nn.Module):
    def __init__(self, n_dim, init_value):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1, n_dim) + init_value)

    def forward(self, input):
        return self.param.repeat(input.size()[0], 1)


class Predictor(BaseProcessingNet):
    def __init__(self, hp, input_size, output_size, num_layers=None, detached=False, spatial=True,
                 final_activation=None, mid_size=None):
        self.spatial = spatial
        mid_size = hp.nz_mid if mid_size is None else mid_size
        if num_layers is None:
            num_layers = hp.n_processing_layers

        super().__init__(input_size, mid_size, output_size, num_layers=num_layers, builder=hp.builder,
                         detached=detached, final_activation=final_activation)

    def forward(self, *inp):
        out = super().forward(*inp)
        return remove_spatial(out, yes=not self.spatial)


class IBPredictor(Predictor):
    """Predictor network with information bottleneck, additionally outputs mean and log_sigma of IB distribution."""
    def __init__(self, hp, input_size, output_size, **kwargs):
        super().__init__(hp, input_size, 2*output_size, **kwargs)   # double output size for Gaussian distribution

    def forward(self, *inp):
        mu, log_sigma = super().forward(*inp).chunk(2, -1)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        return mu + torch.exp(log_sigma) * eps, mu, log_sigma


class Encoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if hp.builder.use_convs:
            self.net = ConvEncoder(hp)
        else:
            self.net = Predictor(hp, hp.state_dim, hp.nz_enc, num_layers=hp.builder.get_num_layers())
    
    def forward(self, input):
        return self.net(input)
        # if self._hp.use_convs and self._hp.use_skips:
        #     return self.net(input)
        # else:
        #     return self.net(input), None
    
    
class ConvEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        
        n = hp.builder.get_num_layers(hp.img_sz)
        self.net = GetIntermediatesSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()
        
        self.net.add_module('input', ConvBlockEnc(in_dim=hp.input_nc, out_dim=hp.ngf, normalization=None,
                                                  builder=hp.builder))
        for i in range(n - 3):
            filters_in = hp.ngf * 2 ** i
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockEnc(in_dim=filters_in, out_dim=filters_in*2, normalize=hp.builder.normalize,
                                             builder=hp.builder))
        
        # add output layer
        self.net.add_module('head', nn.Conv2d(hp.ngf * 2 ** (n - 3), hp.nz_enc, 4))

        self.net.apply(init_weights_xavier)

    def forward(self, input):
        return self.net(input)


class Decoder(nn.Module):
    """ A thin wrapper class that decides which decoder to build """
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        if hp.builder.use_convs:
            assert not (self._hp.add_weighted_pixel_copy & self._hp.pixel_shift_decoder)
            
            if self._hp.pixel_shift_decoder:
                self.net = PixelShiftDecoder(hp)
            elif self._hp.add_weighted_pixel_copy:
                self.net = PixelCopyDecoder(hp)
            else:
                self.net = ConvDecoder(hp)
                
        else:
            assert not self._hp.use_skips
            assert not self._hp.add_weighted_pixel_copy
            assert not self._hp.pixel_shift_decoder
            state_predictor = Predictor(hp, hp.nz_enc, hp.state_dim, num_layers=hp.builder.get_num_layers())
            self.net = AttrDictPredictor({'images': state_predictor})

    def forward(self, input, **kwargs):
        if not (self._hp.pixel_shift_decoder or self._hp.add_weighted_pixel_copy) and 'pixel_source' in kwargs:
            kwargs.pop('pixel_source')
            
        if not self._hp.use_skips and 'skips' in kwargs:
            kwargs.pop('skips')
            
        output = self.net(input, **kwargs)
        return output

    def decode_seq(self, inputs, encodings):
        """ Decodes a sequence of images given the encodings
        
        :param inputs: {'skips', 'pixel_sources'} - pixel_sources is list of source images
        :param encodings:
        :param seq_len:
        :return:
        """
        
        def extend_to_seq(tensor):
            return tensor[:, None].expand([tensor.shape[0], encodings.shape[1]] + list(tensor.shape[1:])).contiguous()

        decoder_inputs = AttrDict(input=encodings)
        if 'skips' in inputs:
            decoder_inputs.skips = map_recursive(extend_to_seq, inputs.skips)
        if 'pixel_sources' in inputs:
            decoder_inputs.pixel_source = map_recursive(extend_to_seq, inputs.pixel_sources)

        return batch_apply(decoder_inputs, self, separate_arguments=True)


class ConvDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self._hp = hp
        n = get_num_conv_layers(hp.img_sz)
        self.net = SkipInputSequential(hp.skips_stride) if hp.use_skips else nn.Sequential()
        out_dim = hp.ngf * 2 ** (n - 3)
        self.net.add_module('net',
                            ConvBlockFirstDec(in_dim=hp.nz_enc, out_dim=out_dim, normalize=hp.builder.normalize,
                                              builder=hp.builder))
        
        for i in reversed(range(n - 3)):
            filters_out = hp.ngf * 2 ** i
            filters_in = filters_out * 2
            if self._hp.use_skips and (i+1) % hp.skips_stride == 0:
                filters_in = filters_in * 2
                
            self.net.add_module('pyramid-{}'.format(i),
                                ConvBlockDec(in_dim=filters_in, out_dim=filters_out, normalize=hp.builder.normalize,
                                             builder=hp.builder))


        self.head_filters_out = filters_out = hp.ngf
        filters_in = filters_out
        if self._hp.use_skips and 0 % hp.skips_stride == 0:
            filters_in = filters_in * 2
            
        self.net.add_module('additional_conv_layer', ConvBlockDec(in_dim=filters_in, out_dim=filters_out,
                                                                      normalization=None, activation=nn.Tanh(), builder=hp.builder))

        self.gen_head = ConvBlockDec(in_dim=filters_out, out_dim=hp.input_nc, normalization=None,
                                                 activation=nn.Tanh(), builder=hp.builder, upsample=False)

        self.net.apply(init_weights_xavier)
        self.gen_head.apply(init_weights_xavier)

    def forward(self, *args, **kwargs):
        output = AttrDict()
        output.feat = self.net(*args, **kwargs)
        output.images = self.gen_head(output.feat)
        return output
    

class PixelCopyDecoder(ConvDecoder):
    def __init__(self, hp, n_masks=None):
        super().__init__(hp)
        self.n_pixel_sources = hp.n_pixel_sources
        n_masks = n_masks or self.n_pixel_sources + 1
        
        self.mask_head = ConvBlockDec(in_dim=self.head_filters_out, out_dim=n_masks,
                                      normalization=None, activation=nn.Softmax(dim=1), builder=hp.builder,
                                      upsample=False)
        self.apply(init_weights_xavier)
        
    def forward(self, *args, pixel_source, **kwargs):
        output = super().forward(*args, **kwargs)
        assert len(pixel_source) == self.n_pixel_sources    # number of pixel sources does not correspond to param

        output.pixel_copy_mask, output.images = self.mask_and_merge(output.feat, pixel_source + [output.images])
        return output
    
    # @torch.jit.script_method
    def mask_and_merge(self, feat, pixel_source):
        # type: (Tensor, List[Tensor]) -> Tuple[Tensor, Tensor]
        
        mask = self.mask_head(feat)
        candidate_images = torch.stack(pixel_source, dim=1)
        images = (mask.unsqueeze(2) * candidate_images).sum(dim=1)
        return mask, images


class PixelShiftDecoder(PixelCopyDecoder):
    def __init__(self, hp):
        self.n_pixel_sources = hp.n_pixel_sources
        super().__init__(hp, n_masks=1 + self.n_pixel_sources * 2)

        self.flow_heads = nn.ModuleList([])
        for i in range(self.n_pixel_sources):
            self.flow_heads.append(ConvBlockDec(in_dim=self.head_filters_out, out_dim=2, normalization=None,
                                                activation=None, builder=hp.builder, upsample=False))
        
        self.apply(init_weights_xavier)
    
    @staticmethod
    def apply_flow(image, flow):
        """ Modified from
        https://github.com/febert/visual_mpc/blob/dev/python_visual_mpc/pytorch/goalimage_warping/goalimage_warper.py#L81
        """

        theta = image.new_tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3).repeat_interleave(image.size()[0], dim=0)
        identity_grid = F.affine_grid(theta, image.size())
        sample_pos = identity_grid + flow.permute(0, 2, 3, 1)
        image = F.grid_sample(image, sample_pos)
        return image
    
    def forward(self, *args, pixel_source, **kwargs):
        output = ConvDecoder.forward(self, *args, **kwargs)
        assert len(pixel_source) == self.n_pixel_sources  # number of pixel sources does not correspond to param

        output.flow_fields = list([head(output.feat) for head in self.flow_heads])
        output.warped_sources = list([self.apply_flow(source, flow) for source, flow in
                                      zip(pixel_source, output.flow_fields)])
        
        _, output.images = self.mask_and_merge(
            output.feat, pixel_source + output.warped_sources + [output.images])
        return output


class Attention(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        time_cond_length = self._hp.max_seq_len if self._hp.one_hot_attn_time_cond else 1
        input_size = hp.nz_enc * 2 + time_cond_length if hp.timestep_cond_attention else hp.nz_enc * 2
        self.query_net = get_predictor(hp, input_size, hp.nz_attn_key)
        self.attention_layers = nn.ModuleList([MultiheadAttention(hp) for _ in range(hp.n_attention_layers)])
        self.predictor_layers = nn.ModuleList([get_predictor(hp, hp.nz_enc, hp.nz_attn_key, num_layers=2)
                                               for _ in range(hp.n_attention_layers)])
        self.out = nn.Linear(hp.nz_enc, hp.nz_enc)

    def forward(self, enc_demo_seq, enc_demo_key_seq, e_l, e_r, start_ind, end_ind, inputs, timestep=None):
        """Performs multi-layered, multi-headed attention."""

        if self._hp.forced_attention:
            return batchwise_index(enc_demo_seq, timestep[:,0].long()), None

        # Get (initial) attention key
        if self._hp.one_hot_attn_time_cond and timestep is not None:
            one_hot_timestep = make_one_hot(timestep.long(), self._hp.max_seq_len).float()
        else:
            one_hot_timestep = timestep
        args = [one_hot_timestep] if self._hp.timestep_cond_attention else []
        
        query = self.query_net(e_l, e_r, *args)
        
        # Attend
        s_ind, e_ind = (torch.floor(start_ind), torch.ceil(end_ind)) if self._hp.mask_inf_attention \
                                                                     else (inputs.start_ind, inputs.end_ind)
        norm_shape_k = query.shape[1:]
        norm_shape_v = enc_demo_seq.shape[2:]
        raw_attn_output, att_weights = None, None
        for attention, predictor in zip(self.attention_layers, self.predictor_layers):
            raw_attn_output, att_weights = attention(query, enc_demo_key_seq, enc_demo_seq, s_ind, e_ind,
                                                     forced_attention_step=timestep if self._hp.forced_attention else None)
            x = F.layer_norm(raw_attn_output, norm_shape_v)
            query = F.layer_norm(predictor(x) + query, norm_shape_k)  # skip connections around attention and predictor

        return apply_linear(self.out, raw_attn_output, dim=1), att_weights     # output non-normalized output of final attention layer


class MultiheadAttention(nn.Module):
    def __init__(self, hp, dropout=0.0):
        super().__init__()
        self._hp = hp
        self.nz = hp.nz_enc
        self.nz_attn_key = hp.nz_attn_key
        self.n_heads = hp.n_attention_heads
        assert self.nz % self.n_heads == 0  # number of attention heads needs to evenly divide latent
        assert self.nz_attn_key % self.n_heads == 0  # number of attention heads needs to evenly divide latent
        self.nz_v_i = self.nz // self.n_heads
        self.nz_k_i = self.nz_attn_key // self.n_heads
        self.temperature = nn.Parameter(self._hp.attention_temperature * torch.ones(1)) if self._hp.learn_attn_temp \
            else self._hp.attention_temperature

        # set up transforms for inputs / outputs
        self.q_linear = nn.Linear(self.nz_attn_key, self.nz_attn_key)
        self.k_linear = nn.Linear(self.nz_attn_key, self.nz_attn_key)
        self.v_linear = nn.Linear(self.nz, self.nz)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.nz, self.nz)

    def forward(self, q, k, v, start_ind, end_ind, forced_attention_step=None):
        batch_size, time = list(k.shape)[:2]
        latent_shape = list(v.shape[2:])

        # perform linear operation and split into h heads
        q = apply_linear(self.q_linear, q, dim=1).view(batch_size, self.n_heads, self.nz_k_i, *latent_shape[1:])
        k = apply_linear(self.k_linear, k, dim=2).view(batch_size, time, self.n_heads, self.nz_k_i, *latent_shape[1:])
        v = apply_linear(self.v_linear, v, dim=2).view(batch_size, time, self.n_heads, self.nz_v_i, *latent_shape[1:])

        # compute masked, multi-headed attention
        vals, att_weights = self.attention(q, k, v, self.nz_k_i, start_ind, end_ind, self.dropout, forced_attention_step)

        # concatenate heads and put through final linear layer
        concat = vals.contiguous().view(batch_size, *latent_shape)
        return apply_linear(self.out, concat, dim=1), att_weights.mean(dim=-1)

    def attention(self, q, k, v, nz_k, start_ind, end_ind, dropout=None, forced_attention_step=None):

        def tensor_product(key, sequence):
            dims = list(range(len(list(sequence.shape)))[3:])
            return (key[:, None] * sequence).sum(dim=dims)

        attn_scores = tensor_product(q, k) / math.sqrt(nz_k) * self.temperature
        attn_scores = MultiheadAttention.mask_out(attn_scores, start_ind, end_ind)
        attn_scores = F.softmax(attn_scores, dim=1)

        if forced_attention_step is not None:
            scores_f = torch.zeros_like(attn_scores)
            batchwise_assign(scores_f, forced_attention_step[:, 0].long(), 1.0)

        scores = scores_f if forced_attention_step is not None else attn_scores
        if dropout is not None and dropout.p > 0.0:
            scores = dropout(scores)

        return (broadcast_final(scores, v) * v).sum(dim=1), attn_scores

    @staticmethod
    def mask_out(scores, start_ind, end_ind):
        # Mask out the frames that are not in the range
        _, mask = mask_out(scores, start_ind, end_ind, -np.inf)
        scores[mask.all(dim=1)] = 1  # When the sequence is empty, fill ones to prevent crashing in Multinomial
        return scores

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if phase == 'train':
            logger.log_scalar(self.temperature, 'attention_softmax_temp', step, phase)


class GaussianPredictor(Predictor):
    def __init__(self, hp, input_dim, gaussian_dim=None, spatial=False):
        if gaussian_dim is None:
            gaussian_dim = hp.nz_vae
            
        super().__init__(hp, input_dim, gaussian_dim * 2, spatial=spatial)
    
    def forward(self, *inputs):
        return Gaussian(super().forward(*inputs)).tensor()


class ApproximatePosterior(GaussianPredictor):
    def __init__(self, hp):
        super().__init__(hp, hp.nz_enc * 3)


class LearnedPrior(GaussianPredictor):
    def __init__(self, hp):
        super().__init__(hp, hp.nz_enc * 2)


class FixedPrior(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

    def forward(self, e_l, *args):  # ignored because fixed prior
        return UnitGaussian([e_l.shape[0], self.hp.nz_vae], self.hp.device).tensor()


class VariationalInference2LayerSharedPQ(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.q1 = GaussianPredictor(hp, hp.nz_enc * 3, hp.nz_vae * 2)
        self.q2 = GaussianPredictor(hp, hp.nz_vae + 2 * hp.nz_enc, hp.nz_vae2 * 2)  # inputs are two parents and z1

    def forward(self, e_l, e_r, e_tilde):
        g1 = self.q1(e_l, e_r, e_tilde)
        z1 = Gaussian(g1).sample()
        g2 = self.q2(z1, e_l, e_r)
        return SequentialGaussian_SharedPQ(g1, z1, g2)


class TwolayerPriorSharedPQ(nn.Module):
    def __init__(self, hp, p1, q_p_shared):
        super().__init__()
        self.p1 = p1
        self.q_p_shared = q_p_shared

    def forward(self, e_l, e_r):
        g1 = self.p1(e_l, e_r)
        z1 = Gaussian(g1).sample()
        g2 = self.q_p_shared(z1, e_l, e_r)  # make sure its the same order of arguments as in usage above!!

        return SequentialGaussian_SharedPQ(g1, z1, g2)


def get_prior(hp):
    if hp.prior_type == 'learned':
        return LearnedPrior(hp)
    elif hp.prior_type == 'fixed':
        return FixedPrior(hp)


def setup_variation_inference(hp):
    if hp.var_inf == '2layer':
        q = VariationalInference2LayerSharedPQ(hp)
        p = TwolayerPriorSharedPQ(hp, get_prior(hp), q.p_q_shared)

    elif hp.var_inf == 'standard':
        q = ApproximatePosterior(hp)
        p = get_prior(hp)

    elif hp.var_inf == 'deterministic':
        q = FixedPrior(hp)
        p = FixedPrior(hp)
        
    return q, p


class SeqEncodingModule(nn.Module):
    def __init__(self, hp, add_time=True):
        super().__init__()
        self.hp = hp
        self.add_time = add_time
        self.build_network(hp.nz_enc + add_time, hp)
        
    def build_network(self, input_size, hp):
        """ This has to define self.net """
        raise NotImplementedError()

    def run_net(self, seq):
        """ Run the network here """
        return self.net(seq)

    def forward(self, seq):
        sh = list(seq.shape)
        seq = seq.view(sh[:2] + [-1])
    
        if self.add_time:
            time = like(torch.arange, seq)(seq.shape[1])[None, :, None].repeat([sh[0], 1, 1])
            seq = torch.cat([seq, time], dim=2)

        proc_seq = self.run_net(seq)
        proc_seq = proc_seq.view(sh)
        return proc_seq


class ConvSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        kernel_size = hp.conv_inf_enc_kernel_size
        assert kernel_size % 2 != 0     # need uneven kernel size for padding
        padding = int(np.floor(kernel_size / 2))
        n_layers = hp.conv_inf_enc_layers
        block = partial(ConvBlock, d=1, kernel_size=kernel_size, padding=padding)
        self.net = BaseProcessingNet(input_size, hp.nz_mid, hp.nz_enc, n_layers, hp.builder, block=block)
        
    def run_net(self, seq):
        # 1d convolutions expect length-last
        proc_seq = self.net(seq.transpose(1, 2)).transpose(1, 2)
        return proc_seq


class RecurrentSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = BaseProcessingLSTM(hp, input_size, hp.nz_enc)


class BidirectionalSeqEncodingModule(SeqEncodingModule):
    def build_network(self, input_size, hp):
        self.net = BidirectionalLSTM(hp, input_size, hp.nz_enc)


class GeneralizedPredictorModel(nn.Module):
    """Predicts the list of output values with optionally different activations."""
    def __init__(self, hp, input_dim, output_dims, activations, detached=False):
        super().__init__()
        assert output_dims  # need non-empty list of output dims defining the number of output values
        assert len(output_dims) == len(activations)     # need one activation for every output dim
        self._hp = hp
        self.activations = activations
        self.output_dims = output_dims
        self.num_outputs = len(output_dims)
        self._build_model(hp, input_dim, detached)

    def _build_model(self, hp, input_dim, detached):
        self.p = Predictor(hp, input_dim, sum(self.output_dims), detached=detached)

    def forward(self, *inputs):
        net_outputs = self.p(*inputs)
        outputs = []
        current_idx = 0
        for output_dim, activation in zip(self.output_dims, self.activations):
            output = net_outputs[:, current_idx:current_idx+output_dim]
            if activation is not None:
                output = activation(output)
            if output_dim == 1:
                output = output.view(-1)      # reduce spatial dimensions for scalars
            outputs.append(output)
        outputs = outputs[0] if len(outputs) == 1 else outputs
        return outputs


class HybridConvMLPEncoder(nn.Module):
    """Encodes image and vector input, fuses features using MLP to produce output feature."""
    def __init__(self, hp):
        super().__init__()
        self._hp = self._default_hparams().overwrite(hp)
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization=self._hp.normalization)

        self._vector_enc = Predictor(self._hp,
                                     input_size=self._hp.input_dim,
                                     output_size=self._hp.nz_enc,
                                     mid_size=self._hp.nz_mid,
                                     num_layers=self._hp.n_layers,
                                     final_activation=None,
                                     spatial=False)
        self._image_enc = Encoder(self._updated_encoder_params())
        self._head = Predictor(self._hp,
                               input_size=2*self._hp.nz_enc,
                               output_size=self._hp.output_dim,
                               mid_size=self._hp.nz_mid,
                               num_layers=2,
                               final_activation=None,
                               spatial=False)

    def _default_hparams(self):
        return ParamDict({
            'input_dim': None,          # dimensionality of the vector input
            'input_res': None,          # resolution of image input
            'output_dim': None,         # dimensionality of output tensor
            'input_nc': 3,              # number of input channels
            'ngf': 8,                   # number of channels in shallowest layer of image encoder
            'nz_enc': 32,               # number of dimensions in encoder-latent space
            'nz_mid': 32,               # number of dimensions for internal feature spaces
            'n_layers': 3,              # number of layers in MLPs
            'normalization': 'none',    # normalization used in encoder network ['none', 'batch']
            'use_convs': False,
            'device': None,
        })

    def forward(self, inputs):
        vector_feature = self._vector_enc(inputs.vector)
        img_feature = remove_spatial(self._image_enc(inputs.image))
        return self._head(torch.cat((vector_feature, img_feature), dim=-1))

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed flat we are not reconstructing
            img_sz=self._hp.input_res,        # image resolution
            input_nc=self._hp.input_nc,       # number of input feature maps
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))


def get_predictor(hp, input_size, output_size, num_layers=None, detached=False):
    """ NOTE: this is deprecated in favor of `Predictor`
    
    :param hp:
    :param input_size:
    :param output_size:
    :param num_layers:
    :param detached:
    :return:
    """
    return Predictor(hp, input_size, output_size, num_layers, detached)


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        return AttrDict()

    def loss(self, *args, **kwargs):
        return {}
