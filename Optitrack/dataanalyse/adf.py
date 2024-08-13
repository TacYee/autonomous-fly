from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
from collections import OrderedDict
from itertools import islice
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from numbers import Number
import numpy as np


def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))


def _normal_log_pdf(value, mu, stddev):
    var = (stddev ** 2)
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - np.log(np.sqrt(2.0*np.pi))


# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))

class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Dropout(nn.Module):
    """ADF implementation of nn.Dropout1d"""
    def __init__(self, p: float = 0.5, keep_variance_fn=None, inplace=False):
        super(Dropout, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.inplace = inplace
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, inputs_mean, inputs_variance):
        if self.training:
            binary_mask = torch.ones_like(inputs_mean)
            binary_mask = F.dropout(binary_mask, self.p, self.training, self.inplace)

            outputs_mean = inputs_mean*binary_mask
            outputs_variance = inputs_variance*binary_mask**2

            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance

        outputs_variance = inputs_variance
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return inputs_mean, outputs_variance
    
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
        super(Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_variance, self.weight**2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance