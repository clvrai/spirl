from functools import partial
import math
import numpy as np
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.nn.parallel._functions import Gather
from torch.optim.optimizer import Optimizer
from torch.nn.modules import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn.functional import interpolate

from spirl.utils.general_utils import batchwise_assign, map_dict, AttrDict, AverageMeter, map_recursive, remove_spatial
from spirl.utils import ndim


class LossSpikeHook:
    def __init__(self, loss_name):
        self.loss_name = loss_name
    
    def run(self, inputs, output, losses, epoch):
        if self.loss_name in losses.keys():
            pass
        
        
class NanGradHook:
    def __init__(self, trainer):
        self.trainer = trainer
        
    def run(self, inputs, output, losses, epoch):
        # TODO fix the scope here
        self.trainer.nan_grads_hook(inputs, output, losses, epoch)


class NoneGradHook:
    def __init__(self, trainer):
        self.trainer = trainer
    
    def run(self, inputs, output, losses, epoch):
        none_list = [n for n, p in filter(lambda x: x[1] is None, self.trainer.model.named_parameters())]
        if none_list: print(none_list)


class RepeatedDataLoader(DataLoader):
    """ A data loader that returns an iterator cycling through data n times """
    def __init__(self, *args, n_repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        if n_repeat != 1:
            self._DataLoader__initialized = False   # this is an ugly hack for pytorch1.3 to be able to change the attr
            self.batch_sampler = RepeatedSampler(self.batch_sampler, n_repeat)
            self._DataLoader__initialized = True
            

class RepeatedSampler(Sampler):
    """ A sampler that repeats the data n times """
    
    def __init__(self, sampler, n_repeat):
        super().__init__(sampler)
        
        self._sampler = sampler
        self.n_repeat = n_repeat
        
    def __iter__(self):
        for i in range(self.n_repeat):
            for elem in self._sampler:
                yield elem

    def __len__(self):
        return len(self._sampler) * self.n_repeat
    
    
def like(func, tensor):
    return partial(func, device=tensor.device, dtype=tensor.dtype)


def make_one_hot(index, length):
    """ Converts indices to one-hot values"""
    oh = index.new_zeros([index.shape[0], length])
    batchwise_assign(oh, index, 1)
    return oh


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return AttrDict()
    
    def loss(self, *args, **kwargs):
        return {}


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class RemoveSpatial(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == 1      # for now we want this to only reduce unary spatial dims
        return remove_spatial(x)


class ResizeSpatial(nn.Module):
    def __init__(self, res, *args, **kwargs):
        super().__init__()
        self._res = res

    def forward(self, x):
        """Assumes input of shape [batch, channel, height, width]"""
        if x.shape[2] != self._res:
            assert True     # for now we don't want this to happen
            return interpolate(x, self._res)
        return x


def batch_cdist(x1, x2, reduction='sum'):
    """ Compute batchwise L2 matrix using quadratic expansion. For each of n vectors in x1, compute L2 norm between it
    and each of m vectors in x2 and outputs the corresponding matrix.
    Adapted from a suggestion somewhere online (look for pytorch github issue comments on cdist).
    
    :param x1: a tensor of shape batchsize x n x dim
    :param x2: a tensor of shape batchsize x m x dim
    :return: a tensor of distances batchsize x n x m
    """
    x1 = x1.flatten(start_dim=2)
    x2 = x2.flatten(start_dim=2)
    
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

    # the einsum is broken, and probably will also be slower
    # torch.einsum('einlhw, eitlhw->nt', torch.stack([x, torch.ones_like(x)]), torch.stack([torch.ones_like(y), y]))
    res = torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)

    # Zero out negative values
    res.clamp_min_(0)
    if reduction == 'mean':
        res = res / x1.shape[2]
    elif reduction == 'sum':
        pass
    else:
        raise NotImplementedError
    return res


def num_parameters(model, level=0):
    """  Returns the number of parameters used in a module.
    
    Known bug: if some of the submodules are repeated, their parameters will be double counted
    :param model:
    :param level: if level==1, returns a dictionary of submodule names and corresponding parameter counts
    :return:
    """
    
    if level == 0:
        return sum([p.numel() for p in model.parameters()])
    elif level == 1:
        return map_dict(num_parameters, dict(model.named_children()))
        
        
class AttrDictPredictor(nn.ModuleDict):
    """ Holds a dictionary of modules and applies them to return an output dictionary """
    def forward(self, *args, **kwargs):
        output = AttrDict()
        for key in self:
            output[key] = self[key](*args, **kwargs)
        return output


def ten2ar(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif np.isscalar(tensor):
        return tensor
    elif hasattr(tensor, 'to_numpy'):
        return tensor.to_numpy()
    else:
        import pdb; pdb.set_trace()
        raise ValueError('input to ten2ar cannot be converted to numpy array')


def ar2ten(array, device, dtype=None):
    if isinstance(array, list) or isinstance(array, dict):
        return array

    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(device)
    else:
        tensor = torch.tensor(array).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def map2torch(struct, device):
    """Recursively maps all elements in struct to torch tensors on the specified device."""
    return map_recursive(partial(ar2ten, device=device, dtype=torch.float32), struct)


def map2np(struct):
    """Recursively maps all elements in struct to numpy ndarrays."""
    return map_recursive(ten2ar, struct)


def avg_grad_norm(model):
    """Computes average gradient norm for the given model."""
    grad_norm = AverageMeter()
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.update(torch.norm(p.grad.data, p=2))
    return grad_norm.avg


def check_shape(t, target_shape):
    if not list(t.shape) == target_shape:
        raise ValueError(f"Temsor should have shape {target_shape} but has shape {list(t.shape)}!")


def mask_out(tensor, start_ind, end_ind, value, dim=1):
    """ Set the elements before start_ind and after end_ind (both inclusive) to the value. """
    if dim != 1:
        raise NotImplementedError
    
    batch_size, time = list(tensor.shape)[:2]
    # (oleg) This creates the indices every time, but doesn't seem to affect the speed a lot.
    inds = torch.arange(time, device=tensor.device, dtype=start_ind.dtype).expand(batch_size, -1)
    mask = (inds >= end_ind[:, None]) | (inds <= start_ind[:, None])
    tensor[mask] = value
    return tensor, mask


class TensorModule(nn.Module):
    """A dummy module that wraps a single tensor and allows it to be handled like a network (for optimizer etc)."""
    def __init__(self, t):
        super().__init__()
        self.t = nn.Parameter(t)

    def forward(self, *args, **kwargs):
        return self.t


def log_sum_exp(tensor, dim=-1):
    """ Safe log-sum-exp operation """
    return torch.logsumexp(tensor, dim)


class DataParallelWrapper(torch.nn.DataParallel):
    """Wraps a pytorch Module for multi-GPU usage but gives access to original model attributes"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def gather(self, outputs, output_device):
        """Overrides the standard gather function to handle custom classes that implement a 'reduce' function."""
        return gather(outputs, output_device, dim=self.dim)


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        try:
            return type(out)(map(gather_map, zip(*outputs)))
        except:
            return type(out).reduce(*outputs)

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def get_padding(seq, replace_dim, size, val=0.0):
    """Returns padding tensor of same shape as seq, but with the target dimension replaced to 'size'.
       All values in returned array are set to 'val'."""
    seq_shape = seq.shape
    if isinstance(seq, torch.Tensor):
        return val * torch.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim+1:], device=seq.device)
    else:
        return val * np.ones(seq_shape[:replace_dim] + (size,) + seq_shape[replace_dim + 1:])


def pad_seq(seq, pre=0, post=0, length=None, val=0.0, dim=1):
    """Pads sequence tensor.
    :arg pre: padded time steps on front
    :arg post: padded time steps on back
    :arg length: maximum length of the padded sequence
    :arg val: value for padded time steps
    :arg dim: dimension in which padding is applied
    """
    if pre > 0:
        seq = ndim.cat((get_padding(seq, dim, pre, val), seq), dim=dim)
    if post > 0:
        seq = ndim.cat((seq, get_padding(seq, dim, post, val)), dim=dim)
    if length is None: return seq
    if seq.shape[dim] < length:
        return ndim.cat((seq, get_padding(seq, dim, length - seq.shape[dim], val)), dim=dim)
    else:
        return ndim.index_select(seq, dim, torch.arange(length))


def stack_with_separator(tensors, dim, sep_width=2):
    """Stacks list of tensors along given dimension, adds separator, brings to range [0...1]."""
    tensors = [(t + 1) / 2 if t.min() < 1e-3 else t for t in tensors]
    stack_tensors = tensors[:1]
    if len(tensors) > 1:
        for tensor in tensors[1:]:
            assert tensor.shape == tensors[0].shape  # all stacked tensors must have same shape!
        separator = get_padding(stack_tensors[0], replace_dim=dim, size=sep_width)
        for tensor in tensors[1:]:
            stack_tensors.extend([separator, tensor])
        stack_tensors = [ndim.cat(stack_tensors, dim=dim)]
    return stack_tensors[0]


class Updater(nn.Module):
    """ A class for non-optimization updates. An Updater defines a 'step' which is called every training step.

    This is implemented as a module so that all updaters that are class fields are easily accessible.
    """

    def __init__(self):
        self.it = 0
        super().__init__()

    def step(self):
        self.it += 1


class ExponentialDecayUpdater(Updater):
    def __init__(self, parameter, n_iter, update_freq=10, min_limit=-np.inf):
        """
        Decays the parameter such that every n_iter the parameter is reduced by 10.

        :param parameter:
        :param n_iter:
        :param update_freq:
        """
        super().__init__()

        assert parameter.numel() == 1
        assert not parameter.requires_grad
        self.parameter = parameter
        self.update_freq = update_freq
        self.min_limit = min_limit

        self.decay = self.determine_decay(n_iter, update_freq)

    def determine_decay(self, n_iter, update_freq):
        n_updates = n_iter / update_freq
        decay = 0.1 ** (1 / n_updates)

        return decay

    def step(self):
        if self.it % self.update_freq == 0 and self.parameter.data[0] * self.decay > self.min_limit:
            self.parameter.data[0] = self.parameter.data[0] * self.decay
        super().step()


class LinearUpdater(Updater):
    def __init__(self, parameter, n_iter, target, update_freq=10, name=None):
        """
        Linearly interpolates the parameter between the current and target value during n_iter iterations

        """
        super().__init__()

        assert parameter.numel() == 1
        assert not parameter.requires_grad
        self.parameter = parameter
        self.update_freq = update_freq
        self.n_iter = n_iter
        self.target = target
        self.name = name

        self.upd = self.determine_upd(n_iter, update_freq, target, parameter.data[0])

    def determine_upd(self, n_iter, update_freq, target, current):
        n_updates = n_iter / update_freq
        upd = (target - current) / n_updates

        return upd

    def step(self):
        if self.it % self.update_freq == 0 and self.it < self.n_iter:
            self.parameter.data[0] = self.parameter.data[0] + self.upd
        super().step()

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if self.name:
            logger.log_scalar(self.parameter, self.name, step, phase)


class ConstantUpdater(Updater):
    def __init__(self, parameter, n_iter, name=None):
        """
        Keeps the parameter constant for n_iter
        """
        super().__init__()

        assert parameter.numel() == 1
        # assert not parameter.requires_grad
        self.parameter = parameter
        self.n_iter = n_iter
        self.name = name
        self.val = parameter.data.clone()

    def step(self):
        # TODO this should depend on the global step
        if self.it < self.n_iter:
            self.parameter.data = self.val.clone().to(self.parameter.device)

        super().step()

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if self.name:
            logger.log_scalar(self.parameter, self.name, step, phase)


def get_constant_parameter(init_log_value, learnable):
    return torch.nn.Parameter(torch.full((1,), init_log_value)[0], requires_grad=learnable)


def sample_idx_batch_in_range(start, end, batch_size=None, device=None):
    """Samples batch of indices in range [start, end). Uses batch dimension of start/end if not scalars,
    else can feed separate batch dimension."""
    # figure out batch size
    if not np.isscalar(start):
        bs = start.shape[0]
    if not np.isscalar(end):
        bs = end.shape[0]
    if batch_size is not None:
        bs = batch_size

    # bring all arrays to proper size
    if np.isscalar(start):
        start = start * torch.ones((bs,), device=device)
    if np.isscalar(end):
        end = end * torch.ones((bs,), device=device)

    # sample indices - range does not include end
    idxs = torch.rand((bs,), device=device) * (end.float() - start.float() - 1) + start.float()
    return idxs.long()


def remove_grads(module):
    """Sets requires_grad for all params in module to False."""
    for p in module.parameters():
        p.requires_grad = False


def switch_off_batchnorm_update(model):
    """Switches off batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.eval()


def switch_on_batchnorm_update(model):
    """Switches on batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.train()


@contextmanager
def no_batchnorm_update(model):
    """Switches off all batchnorm updates within context."""
    switch_off_batchnorm_update(model)
    yield
    switch_on_batchnorm_update(model)


def find(inp, success_fn):
    """ Finds an element for which the success_fn responds true """

    def rec_find(structure):
        if isinstance(structure, list) or isinstance(structure, tuple):
            items = list(structure)
        elif isinstance(structure, dict):
            items = list(structure.items())
        else:
            # Base case, if not dict or iterable
            success = success_fn(structure)
            return structure, success

        # If dict or iterable, iterate and find a tensor
        for item in items:
            result, success = rec_find(item)
            if success:
                return result, success

        return None, False

    return rec_find(inp)[0]


def find_tensor(structure, min_dim=None):
    """ Finds a single tensor in the structure """

    def success_fn(x):
        success = isinstance(x, torch.Tensor)
        if min_dim is not None:
            success = success and len(x.shape) >= min_dim

        return success

    return find(structure, success_fn)


def update_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # test stacking+padding for tensors/np_arrays
    def test_stack_pad(generalized_tensors):
        generalized_tensors = [pad_seq(t, pre=1, length=10) for t in generalized_tensors]
        stacked_t = stack_with_separator(generalized_tensors, dim=3)
        print(stacked_t.shape)
    test_stack_pad([np.random.rand(5, 8, 3, 32, 32) for _ in range(3)])
    test_stack_pad([torch.rand(5, 8, 3, 32, 32) for _ in range(3)])

    # test decay updater
    x = torch.ones(1)
    upd = ExponentialDecayUpdater(x, 100, 2)
    
    for i in range(1000):
        upd.step()
        
    print(x)
