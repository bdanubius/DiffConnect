import torch

import torch.autograd

from torch.nn import Parameter, Module, Sequential, Linear, ReLU
from torch.nn.functional import mse_loss
from collections import OrderedDict
from utils import count_parameters

import numpy as np


def fix_name(name):
    return name.replace('.', '_')


class HyperNetwork(Module):

    def __init__(self, target_net):
        super().__init__()
        self.last_hyper_tensor = None
        target_params = OrderedDict(target_net.named_parameters())
        self.orig_param_count = count_parameters(target_net)
        self.param_names = [fix_name(name) for name in target_params.keys()]
        self.lambdas = [self.make_hyper_lambda(fix_name(name), param) for name, param in target_params.items()]
        self.outputs = None
        self.__dict__['target_params'] = list(target_params.values())
        # ^ use of __dict__ stops us from owning the params

    def __post_init__(self):
        print(f"Creating {type(self).__name__} (modelling {self.orig_param_count} with {count_parameters(self)} params)")

    def forward(self, set_arrays=True):
        self.outputs = [fn() for fn in self.lambdas]
        if set_arrays:
            with torch.no_grad():
                return [dst.copy_(src) for src, dst in zip(self.outputs, self.target_params)]
        return self.outputs

    def backward(self):
        grad_tensors = [p.grad for p in self.target_params]
        torch.autograd.backward(self.outputs, grad_tensors)

    def discrepancy_loss(self):
        loss = sum(mse_loss(src, dst.detach(), reduction='sum')
                   for src, dst in zip(self.outputs, self.target_params))
        return loss

    def make_hyper_lambda(self, name, param):
        raise NotImplementedError()

    def make_hyper_tensor(self, name, shape, var=1.0, set_last=True, fix=False):
        param = torch.randn(shape) * np.sqrt(var)
        param_name = 'hyper_' + name
        if set_last: self.last_hyper_tensor = param_name
        if not fix:
            param = Parameter(param)
            self.register_parameter(param_name, param)
        return param

    def get_last_hyper_tensor(self):
        return self._parameters[self.last_hyper_tensor]


class DummyHyperNetwork(HyperNetwork):

    def __init__(self, target_net):
        super().init(target_net)
        self.__post_init__()

    def make_hyper_lambda(self, name, param):
        param = self.make_hyper_tensor(name, param.shape)
        return lambda: param

    def absorb(self, step_size=0.1):
        with torch.no_grad():
            for dst, name in zip(self.target_params, self.param_names):
                src = self._parameters['hyper_' + name]
                src.copy_(src * (1-step_size) + dst * step_size)


class RandomBasisHyperNetwork(HyperNetwork):

    def __init__(self, target_net, ndims=5):
        self.ndims = ndims
        super().__init__(target_net)
        coeffs = torch.zeros((ndims, ))
        coeffs[0] = 1.0
        self.coeffs = Parameter(coeffs)
        self.__post_init__()

    def make_hyper_lambda(self, name, param):
        n = self.ndims
        shape = list(param.shape)
        var = param.var().item()
        param = self.make_hyper_tensor(name, [n] + shape, fix=True, var=var)
        return lambda: torch.matmul(self.coeffs, param.view(n, -1)).view(*shape)

