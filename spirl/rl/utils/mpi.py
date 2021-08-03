"""Helpers for multi-processing."""

from mpi4py import MPI
import signal
import sys
import numpy as np
import torch

from spirl.utils.general_utils import AttrDict, joinListDictList


def update_with_mpi_config(conf):
    mpi_config = AttrDict()
    rank = MPI.COMM_WORLD.Get_rank()
    mpi_config.rank = rank
    mpi_config.is_chef = rank == 0
    mpi_config.num_workers = MPI.COMM_WORLD.Get_size()
    conf.mpi = mpi_config

    # update conf
    conf.general.seed = conf.general.seed + rank
    return conf


def set_shutdown_hooks():
    def shutdown(signal, frame):
        print('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def mpi_sum(x):
    buf = np.zeros_like(np.array(x))
    MPI.COMM_WORLD.Allreduce(np.array(x), buf, op=MPI.SUM)
    return buf


# sync gradients across the different cores
def sync_grads(network):
    flat_grads, grads_shape = _get_flat_grads(network)
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_grads(network, grads_shape, global_grads)


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)


def mpi_gather_experience_batch(experience_batch):
    buf = MPI.COMM_WORLD.allgather(experience_batch)
    return joinListDictList(buf)


def mpi_gather_experience(experience_batch):
    """Gathers data across workers, can handle hierarchical and flat experience dicts."""
    if hasattr(experience_batch, 'hl_batch'):
        # gather HL and LL batch separately
        return AttrDict(
            hl_batch=mpi_gather_experience_batch(experience_batch.hl_batch),
            ll_batch=mpi_gather_experience_batch(experience_batch.ll_batch),
        )
    else:
        return mpi_gather_experience_batch(experience_batch)


def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        if not value.requires_grad or value.grad is None: continue
        try:
            grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        except:
            print('Cannot get grad of tensor {}'.format(key_name))
            import pdb; pdb.set_trace()
        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape


def _set_flat_grads(network, grads_shape, flat_grads):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, value in network.named_parameters():
        if not value.requires_grad or value.grad is None: continue
        len_grads = int(np.prod(grads_shape[key_name]))
        copy_grads = flat_grads[pointer:pointer + len_grads].reshape(grads_shape[key_name])
        copy_grads = torch.tensor(copy_grads).to(device)
        # copy the grads
        value.grad.data.copy_(copy_grads.data)
        pointer += len_grads


# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape


# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = int(np.prod(params_shape[key_name]))
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param
