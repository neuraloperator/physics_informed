import os
import torch
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '7777'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return dist.get_world_size()


def all_reduce_mean(tensor):
    '''
    Reduce the tensor across all machines, the operation is in-place.
    :param tensor: tensor to reduce
    :return: reduced tensor
    '''
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.div_(world_size)


def reduce_sum(tensor):
    '''
    Reduce the tensor across all machines. Only process with rank 0 will receive the final result
    Args:
        tensor: input and ouput of the collective. The function operates in-place
    Returns:
        final result
    '''
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    return tensor


def reduce_loss_dict(loss_dict):
    if not dist.is_available() or dist.is_initialized():
        return loss_dict
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses