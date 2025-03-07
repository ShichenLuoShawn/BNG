import torch
import numpy as np
from random import randrange


def get_fc(timeseries, sampling_point, window_size, self_loop):
    fc = corrcoef(timeseries[sampling_point:sampling_point+window_size].T)
    if not self_loop: fc-= torch.eye(fc.shape[0])
    return fc

def get_minibatch_fc(minibatch_timeseries, sampling_point, window_size, self_loop):
    fc_list = []
    for timeseries in minibatch_timeseries:
        fc = get_fc(timeseries, sampling_point, window_size, self_loop)
        fc_list.append(fc)
    return torch.stack(fc_list)


def process_dynamic_fc(minibatch_timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = minibatch_timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert minibatch_timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert minibatch_timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(minibatch_timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    minibatch_fc_list = [get_minibatch_fc(minibatch_timeseries, sampling_point, window_size, self_loop) for sampling_point in sampling_points]
    dynamic_fc = torch.stack(minibatch_fc_list, dim=1)

    return dynamic_fc, sampling_points


# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c)+1e-9)
    c = c.div(stddev.expand_as(c).t()+1e-9)
    c = torch.clamp(c, -1.0, 1.0)
    return c
