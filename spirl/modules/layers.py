import math
from functools import partial

import torch.nn as nn
from spirl.utils.general_utils import ConcatSequential, HasParameters
from spirl.utils.general_utils import AttrDict
from torch.nn import init


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        pass    # by default PyTorch uses Kaiming_Normal initializer


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class Block(nn.Sequential, HasParameters):
    def __init__(self, **kwargs):
        self.builder = kwargs.pop('builder')
        nn.Sequential.__init__(self)
        HasParameters.__init__(self, **kwargs)
        if self.params.normalization is None or self.params.normalization == 'none':
            self.params.normalize = False
        if not self.params.normalize:
            self.params.normalization = None
        
        self.build_block()
        self.complete_block()
        
    def get_default_params(self):
        params = AttrDict(
            normalize=True,
            activation=nn.LeakyReLU(0.2, inplace=True),
            normalization=self.builder.normalization,
            normalization_params=AttrDict()
        )
        return params
        
    def build_block(self):
        raise NotImplementedError
    
    def complete_block(self):
        if self.params.normalization is not None:
            self.params.normalization_params.affine = True
            # TODO add a warning if the normalization is over 1 element
            if self.params.normalization == 'batch':
                normalization = nn.BatchNorm1d if self.params.d == 1 else nn.BatchNorm2d
                self.params.normalization_params.track_running_stats = True
                
            elif self.params.normalization == 'instance':
                normalization = nn.InstanceNorm1d if self.params.d == 1 else nn.InstanceNorm2d
                self.params.normalization_params.track_running_stats = True
                # TODO if affine is false, the biases will not be learned
                
            elif self.params.normalization == 'group':
                normalization = partial(nn.GroupNorm, 8)
                if self.params.out_dim < 32:
                    raise NotImplementedError("note that group norm is likely to not work with this small groups")
                
            else:
                raise ValueError("Normalization type {} unknown".format(self.params.normalization))
            self.add_module('norm', normalization(self.params.out_dim, **self.params.normalization_params))

        if self.params.activation is not None:
            self.add_module('activation', self.params.activation)
            
###

class FCBlock(Block):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            d=1,
        ))
        return params
    
    def build_block(self):
        self.add_module('linear', nn.Linear(self.params.in_dim, self.params.out_dim, bias=not self.params.normalize))


class ConvBlock(Block):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            d=2,
            kernel_size=3,
            stride=1,
            padding=1,
        ))
        return params
        
    def build_block(self):
        if self.params.d == 2:
            cls = nn.Conv2d
        elif self.params.d == 1:
            cls = nn.Conv1d
            
        self.add_module('conv', cls(
            self.params.in_dim, self.params.out_dim, self.params.kernel_size, self.params.stride, self.params.padding,
            bias=not self.params.normalize))


class ConvBlockEnc(ConvBlock):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            kernel_size=4,
            stride=2,
        ))
        return params
        

class ConvBlockDec(ConvBlock):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            activation=nn.ReLU(True),
            kernel_size=4,
            stride=1,
            padding=0,
            upsample=True,
            asym_padding=(1, 2, 1, 2),
        ))
        return params
    
    def build_block(self):
        if self.params.upsample:
            self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))

        if self.params.asym_padding:
            self.add_module('pad', nn.ZeroPad2d(padding=self.params.asym_padding))
            
        self.add_module('conv', nn.Conv2d(
            self.params.in_dim, self.params.out_dim, self.params.kernel_size, self.params.stride, self.params.padding,
            bias=not self.params.normalize))
        

class ConvBlockFirstDec(ConvBlockDec):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            kernel_size=4,
            stride=1,
            padding=0,
            upsample=False,
        ))
        return params

    def build_block(self):
        if self.params.upsample:
            self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
    
        self.add_module('conv', nn.ConvTranspose2d(
            self.params.in_dim, self.params.out_dim, self.params.kernel_size, self.params.stride, self.params.padding,
            bias=not self.params.normalize))

###


class BaseProcessingNet(ConcatSequential):
    """ Constructs a network that keeps the activation dimensions the same throughout the network
    Builds an MLP or CNN, depending on the builder. Alternatively uses custom blocks """
    
    def __init__(self, in_dim, mid_dim, out_dim, num_layers, builder, block=None, detached=False,
                 final_activation=None):
        super().__init__(detached)

        if block is None:
            block = builder.def_block
        block = builder.wrap_block(block)
        
        self.add_module('input', block(in_dim=in_dim, out_dim=mid_dim, normalization=None))
        for i in range(num_layers):
            self.add_module('pyramid-{}'.format(i),
                            block(in_dim=mid_dim, out_dim=mid_dim, normalize=builder.normalize))

        self.add_module('head'.format(i + 1),
                        block(in_dim=mid_dim, out_dim=out_dim, normalization=None, activation=final_activation))
        self.apply(init_weights_xavier)
        
        
def get_num_conv_layers(img_sz):
    n = math.log2(img_sz)
    assert n == round(n), 'imageSize must be a power of 2'
    assert n >= 3, 'imageSize must be at least 8'
    return int(n)


class LayerBuilderParams:
    """ This class holds general parameters for all subnetworks, such as whether to use convolutional networks, etc """
    
    def __init__(self, use_convs, normalization='batch'):
        self.use_convs = use_convs
        self.init_fn = init_weights_xavier
        self.normalize = normalization != 'none'
        self.normalization = normalization

    @property
    def get_num_layers(self):
        if self.use_convs:
            return get_num_conv_layers
        else:
            return lambda: 2

    @property
    def def_block(self):
        """ Fetches the default block to use"""
        if self.use_convs:
            return ConvBlock
        else:
            return FCBlock
            
    def wrap_block(self, block):
        """ Wraps a block with the builder defaults. This function needs to be used every time a block is created. """
        # TODO fix this up. The blocks can do this.
        return partial(block, builder=self, normalization=self.normalization)

