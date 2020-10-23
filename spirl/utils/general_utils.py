import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from contextlib import contextmanager
import heapq
import inspect
import time
import copy
import re
import random
import functools
from functools import partial, reduce
import collections
from collections import OrderedDict

from spirl.utils import ndim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg


class AverageTimer(AverageMeter):
    """Times whatever is inside the with self.time(): ... block, exposes average etc like AverageMeter."""
    @contextmanager
    def time(self):
        self.start = time.time()
        yield
        self.update(time.time() - self.start)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]


def add_n_dims(generalized_tensor, n, dim=-1):
    """ Adds n new dimensions of size 1 to the end of the tensor or array """
    for i in range(n):
        generalized_tensor = ndim.unsqueeze(generalized_tensor, dim)
    return generalized_tensor


def broadcast_final(t1, t2):
    """ Adds trailing dimensions to t1 to match t2 """
    return add_n_dims(t1, len(t2.shape) - len(t1.shape))


class MiniTrainer:
    def __init__(self, model=None, step_fn=None, parameters=None):
        """

        :param model: Either model or parameters have to be specified
        :param step_fn:
        :param parameters: Either model or parameters have to be specified
        """
        if parameters is None:
            parameters = model.parameters()

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=0.001)
        self.step_fn = step_fn

    def step(self, i):
        self.optimizer.zero_grad()
        loss = self.step_fn(i)
        loss.backward()
        if i % 1 == 0: print(loss)
        self.optimizer.step()

    def train(self, time):
        [self.step(i) for i in range(time)]


def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


@contextmanager
def dummy_context():
    yield


def get_clipped_optimizer(*args, optimizer_type=None, **kwargs):
    assert optimizer_type is not None  # need to set optimizer type!

    class ClipGradOptimizer(optimizer_type):
        def __init__(self, *args, gradient_clip=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.gradient_clip = gradient_clip

        def step(self, *args, **kwargs):
            if self.gradient_clip is not None:
                params = np.concatenate([group['params'] for group in self.param_groups])
                torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)

            super().step(*args, **kwargs)

    return ClipGradOptimizer(*args, **kwargs)


class optional:
    """ A function decorator that returns the first argument to the function if yes=False
     I chose a class-based decorator since I find the syntax less confusing. """

    def __init__(self, n=0):
        """ Decorator parameters """
        self.n = n

    def __call__(self, func):
        """ Wrapping """

        def wrapper(*args, yes=True, **kwargs):
            if yes:
                return func(*args, **kwargs)

            n = self.n
            if inspect.ismethod(func):
                n += 1
            return args[n]

        return wrapper


@contextmanager
def timing(text, name=None, interval=10):
    start = time.time()
    yield
    elapsed = time.time() - start

    if name:
        if not hasattr(timing, name):
            setattr(timing, name, AverageMeter())
        meter = getattr(timing, name)
        meter.update(elapsed)
        if meter.count % interval == 0:
            print("{} {}".format(text, meter.avg))
        return

    print("{} {}".format(text, elapsed))


class timed:
    """ A function decorator that prints the elapsed time """

    def __init__(self, text):
        """ Decorator parameters """
        self.text = text

    def __call__(self, func):
        """ Wrapping """

        def wrapper(*args, **kwargs):
            with timing(self.text):
                result = func(*args, **kwargs)
            return result

        return wrapper


def get_dim_inds(generalized_tensor):
    """ Returns a tuple 0..length, where length is the number of dimensions of the tensors"""
    return tuple(range(len(generalized_tensor.shape)))


def dict_concat(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        raise ValueError("Dict keys are not equal. got {} vs {}.".format(d1.keys(), d2.keys()))
    for key in d1:
        d1[key] = np.concatenate((d1[key], d2[key]))


def float_regex():
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    return rx


def lazy_property(function):
    """ Caches the property such that the code creating it is only executed once.
    Adapted from Dani Hafner (https://danijar.com/structuring-your-tensorflow-models/) """
    # TODO can I just use lru_cache?
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


def rand_split_list(list, frac=0.5, seed=None):
    rng = random.Random()
    if seed is not None: rng.seed(seed)
    rng.shuffle(list)
    split = int(frac * len(list))
    return list[:split], list[split:]


class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)


def batch_apply(tensors, fn, separate_arguments=False, unshape_inputs=False, ):
    """ Applies the fn to the tensors while treating two first dimensions of tensors as batch.

    :param tensors: can be a single tensor, tuple or a list.
    :param fn: _fn_ can return a single tensor, tuple or a list
    :param separate_arguments: if true, the highest-level list will be fed into the function as a
    list of arguments
    :param unshape_inputs: if true, reshapes the inputs back to original (in case they have references to classes)"""
    
    reference_tensor, success = find_tensor(tensors, min_dim=2)
    if not success:
        raise ValueError("couldn't find a reference tensor")
    
    batch, time = reference_tensor.shape[:2]
    reshape_to = make_recursive(lambda tensor: tensor.view([batch * time] + list(tensor.shape[2:])))
    reshape_from = make_recursive(lambda tensor: tensor.view([batch, time] + list(tensor.shape[1:])))
    
    input_reshaped = reshape_to(tensors)
    if separate_arguments:
        if isinstance(input_reshaped, dict):
            output = fn(**input_reshaped)
        else:
            output = fn(*input_reshaped)
    else:
        output = fn(input_reshaped)
    output_reshaped_back = reshape_from(output)
    if unshape_inputs: reshape_from(input_reshaped)
    return output_reshaped_back


def find_tensor(tensors, min_dim=None):
    """ Finds a single tensor in the structure """

    if isinstance(tensors, list) or isinstance(tensors, tuple):
        tensors_items = list(tensors)
    elif isinstance(tensors, dict):
        tensors_items = list(tensors.items())
    else:
        # Base case, if not dict or iterable
        success = isinstance(tensors, torch.Tensor)
        if min_dim is not None: success = success and len(tensors.shape) >= min_dim
        return tensors, success

    # If dict or iterable, iterate and find a tensor
    for tensors_item in tensors_items:
        tensors_result, success = find_tensor(tensors_item)
        if success:
            return tensors_result, success

    return None, False


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """
    
    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor) or isinstance(tensors, np.ndarray):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))
    
    return recursive_map


def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


recursively = make_recursive


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)


def batchwise_index(generalized_tensor, index, dim=1):
    """ Indexes the tensor with the _index_ along dimension dim.
    Works for numpy arrays and torch tensors
    
    :param generalized_tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 such that t2[i] = tensor[i,index[i]]
    """
        
    bs = generalized_tensor.shape[0]
    return ndim.swapaxes(generalized_tensor, 1, dim)[np.arange(bs), index]


def batchwise_assign(tensor, index, value):
    """ Assigns the _tensor_ elements at the _index_ the _value_. The indexing is along dimension 1

    :param tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 where that t2[i, index[i]] = value
    """
    bs = tensor.shape[0]
    tensor[np.arange(bs), index] = value


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


def subdict(dict, keys, strict=True):
    if not strict:
        keys = dict.keys() & keys
    return AttrDict((k, dict[k]) for k in keys)


def split_along_axis(array, axis):
    """Splits array along axis into single-dimension elements. Returns list (removes split axis)."""
    sarray = np.split(array, array.shape[axis], axis)
    return [elem.squeeze(axis) for elem in sarray]


def obj2np(obj):
    """Wraps an object into an np.array."""
    ar = np.zeros((1,), dtype=np.object_)
    ar[0] = obj
    return ar


def np2obj(np_array):
    if isinstance(np_array, list) or np_array.size > 1:
        return [e[0] for e in np_array]
    else:
        return np_array[0]


@optional()
def remove_spatial(tensor):
    if len(tensor.shape) == 4 or len(tensor.shape) == 5:
        return tensor.mean(dim=[-1, -2])
    elif len(tensor.shape) == 2:
        return tensor
    else:
        raise ValueError("Are you sure you want to do this? Got tensor shape {}".format(tensor.shape))


def apply_linear(layer, val, dim):
    """Applies a liner layer to the specified dimension."""
    assert isinstance(layer, nn.Linear)     # can only apply linear layers
    return layer(val.transpose(dim, -1)).transpose(dim, -1)


def nan_hook(tensor_struct):
    """Jumps into pdb if NaN is detected in tensor_struct.
    tensor_struct can be any nested list/dict structure containing tensors."""
    def check_nan(t):
        if (isinstance(t, torch.Tensor) and torch.isnan(t).any()) \
                or (isinstance(t, np.ndarray) and np.isnan(t).any()):
            import pdb; pdb.set_trace()
    make_recursive(check_nan)(tensor_struct)


class GetIntermediatesSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, input):
        """Computes forward pass through the network outputting all intermediate activations with final output."""
        skips = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)
            
            if i % self.stride == 0:
                skips.append(input)
            else:
                skips.append(None)
        return input, skips[:-1]


class SkipInputSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        
    def forward(self, input, skips):
        """Computes forward pass through the network and concatenates input skips."""
        skips = [None] + skips[::-1]   # start applying skips after the first decoding layer
        for i, module in enumerate(self._modules.values()):
            if i < len(skips) and skips[i] is not None:
                input = torch.cat((input, skips[i]), dim=1)
                
            input = module(input)
        return input


class ConcatSequential(nn.Sequential):
    """ A sequential net that accepts multiple arguments and concatenates them along dimension 1
    The class also broadcasts the tensors to fill last dimensions.
    """
    def __init__(self, detached=False):
        super().__init__()
        self.detached = detached
    
    def forward(self, *inp):
        inp = concat_inputs(*inp)
        if self.detached:
            inp = inp.detach()
        return super().forward(inp)


def concat_inputs(*inp):
    """ Concatenates tensors together. Used if the tensors need to be passed to a neural network as input. """
    max_n_dims = np.max([len(tensor.shape) for tensor in inp])
    inp = torch.cat([add_n_dims(tensor, max_n_dims - len(tensor.shape)) for tensor in inp], dim=1)
    return inp


def select_e_0_e_g(seq, start_ind, end_ind):
    e_0 = batchwise_index(seq, start_ind)
    e_g = batchwise_index(seq, end_ind)
    return e_0, e_g


def shuffle_with_seed(arr, seed=1):
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(arr)
    return arr


def interleave_lists(*args):
    """Interleaves N lists of equal length."""
    for l in args:
        assert len(l) == len(args[0])      # all lists need to have equal length
    return [val for tup in zip(*args) for val in tup]


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})


def get_end_ind(pad_mask):
    """
    :param pad_mask: torch tensor with 1 where there is an actual image and zeros where there's padding
    pad_mask has shape batch_size x max_seq_len
    :return:
    """
    max_seq_len = pad_mask.shape[1]
    end_ind = torch.argmax(pad_mask * torch.arange(max_seq_len, dtype=torch.float, device=pad_mask.device), 1)

    return end_ind


def get_pad_mask(end_ind, max_seq_len):
    """
    :param pad_mask: torch tensor with 1 where there is an actual image and zeros where there's padding
    pad_mask has shape batch_size x max_seq_len
    :return:
    """
    use_torch = isinstance(end_ind, torch.Tensor)
    
    if use_torch:
        arange_fn = partial(torch.arange, dtype=torch.long, device=end_ind.device)
    else:
        arange_fn = np.arange
    
    pad_mask = arange_fn(max_seq_len) <= end_ind[:, None]
    
    if use_torch:
        pad_mask = pad_mask.float()
    else:
        pad_mask = pad_mask.astype(np.float)
    
    return pad_mask


class HasParameters:
    def __init__(self, **kwargs):
        self.build_params(kwargs)
        
    def build_params(self, inputs):
        # If params undefined define params
        try:
            self.params
        except AttributeError:
            self.params = self.get_default_params()
            self.params.update(inputs)
    
    # TODO allow to access parameters by self.<param>
    

def maybe_retrieve(d, key):
    if hasattr(d, key):
        return d[key]
    else:
        return None


class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self


class Schedule:
    """Container for parameter schedules."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        return ParamDict({})

    def __call__(self, t):
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, config):
        super().__init__(config)
        self._p = self._hp.p

    def _default_hparams(self):
        return super()._default_hparams().overwrite(AttrDict(
            p=None
        ))

    def __call__(self, t):
        return self._p


class LinearSchedule(Schedule):
    def __init__(self, config):
        """Linear interpolation between initial_p and final_p over schedule_timesteps."""
        super().__init__(config)
        self.schedule_timesteps = self._hp.schedule_timesteps
        self.final_p = self._hp.final_p
        self.initial_p = self._hp.initial_p

    def _default_hparams(self):
        return super()._default_hparams().overwrite(AttrDict(
            initial_p=None,
            final_p=None,
            schedule_timesteps=None,
        ))

    def __call__(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class DelayedLinearSchedule(LinearSchedule):
    def _default_hparams(self):
        return super()._default_hparams().overwrite(AttrDict(
            delay=None,
        ))

    def __call__(self, t):
        if t < self._hp.delay:
            return super().__call__(0)
        else:
            return super().__call__(t - self._hp.delay)


class DictFlattener:
    """Flattens all elements in ordered dict into single vector, remembers structure and can unflatten back."""
    def __init__(self):
        self._example_struct = None

    def __call__(self, d):
        """Flattens dict d into vector."""
        assert isinstance(d, OrderedDict)
        if self._example_struct is None:
            self._example_struct = copy.deepcopy(d)
        assert d.keys() == self._example_struct.keys()
        return np.concatenate([d[key] for key in d])

    def unflatten(self, v):
        """Restores original dict structure."""
        output, idx = OrderedDict(), 0
        for key in self._example_struct:
            output[key] = v[idx : idx + self._example_struct[key].shape[0]]
            idx += self._example_struct[key].shape[0]
        return output


def pretty_print(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key) + ':')
            pretty_print(value, indent+1)
        else:
            print('\t' * indent + str(key) + ':' + '\t' + str(value))
