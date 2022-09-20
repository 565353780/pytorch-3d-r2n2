#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import collections
import numpy as np


#utility function to check nan
def has_nan(x):
    """
    x is a torch tensor. (x != x) will return a torch.ByteTensor whose 
    elements are either 1 or 0. And (x != x).any() will return True if
    any elements in the tensor are non-zero. Note that (nan != nan) is 
    True. If there is any nan in x, then the function will return True.
    """
    return (x != x).any()


def max_or_nan(params):
    params = list(params)
    nan_or_max_param = torch.FloatTensor(len(params)).zero_()
    if torch.cuda.is_available():
        nan_or_max_param = nan_or_max_param.cuda()

    for param_idx, param in enumerate(params):
        # If there is nan, max will return nan
        # Note that param is Variable
        nan_or_max_param[param_idx] = torch.max(torch.abs(param)).item()
        # print('param %d : %f' % (param_idx, nan_or_max_param[param_idx]))
    return nan_or_max_param


#utility function to customize weight initialization
def weight_init(w_shape, mean=0, std=0.01, filler='msra'):
    rng = np.random.RandomState()
    if isinstance(w_shape, collections.Iterable):
        if len(w_shape) > 1 and len(w_shape) < 5:
            fan_in = np.prod(w_shape[1:])
            fan_out = np.prod(w_shape) / w_shape[1]
            n = (fan_in + fan_out) / 2.
        elif len(w_shape) == 5:
            # 3D Convolution filter
            fan_in = np.prod(w_shape[1:])
            fan_out = np.prod(w_shape) / w_shape[2]
            n = (fan_in + fan_out) / 2.
        else:
            raise NotImplementedError(
                'Filter shape with ndim > 5 not supported: len(w_shape) = %d' %
                len(w_shape))
    else:
        raise Exception(
            "w_shape should be an instance of collections.Iterable")

    if filler == 'gaussian':
        np_values = np.asarray(rng.normal(mean, std, w_shape))
    elif filler == 'msra':
        np_values = np.asarray(rng.normal(mean, np.sqrt(2. / n), w_shape))
    elif filler == 'xavier':
        scale = np.sqrt(3. / n)
        np_values = np.asarray(
            rng.uniform(low=-scale, high=scale, size=w_shape))
    elif filler == 'constant':
        np_values = mean * np.ones(w_shape)
    elif filler == 'orth':
        ndim = np.prod(w_shape)
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        np_values = u.reshape(w_shape)
    else:
        raise NotImplementedError('Filler %s not implemented' % filler)
    torch_tensor = torch.from_numpy(np_values).type(torch.FloatTensor)
    return torch_tensor
