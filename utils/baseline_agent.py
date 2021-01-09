import torch
import torch.nn as nn
from utils.utils import find_activation, Flatten, Nomramlization


class Base_Agent(nn.Module):

    def __init__(self):
        super(Base_Agent, self).__init__()

    def build_model(self):
        pass

    def forward(self, state, *args):
        pass

    def loss(self, *args):
        pass


class MLP(nn.Module):

    def __init__(
        self,
        init_size=1,
        num_of_layers=1,
        num_of_unit=32,
        activation='relu',
        **kwargs
    ):

        super(MLP, self).__init__()
        self.batch_normal = False
        self.normal = False

        if 'normal' in kwargs.keys():
            self.normal = kwargs['normal']

        if 'batch_norm' in kwargs.keys():
            self.batch_normal = kwargs['batch_norm']
        self.init_size = init_size
        self.num_of_layers = num_of_layers
        if type(num_of_unit) == int:
            num_of_unit = [num_of_unit]
        self.num_of_unit = list(num_of_unit)
        self.activation = activation
        self.module = self.build_model()

    def build_model(self):

        model = nn.Sequential()
        init_size = self.init_size
        if self.normal:
            model.add_module('normal', self.normal)
        for i in range(self.num_of_layers):
            model.add_module('layer_'+str(i), torch.nn.Linear(init_size, self.num_of_unit[i]))
            init_size = self.num_of_unit[i]

            if self.batch_normal:
                model.add_module("batch_norm_"+str(i), nn.BatchNorm1d(self.num_of_unit[i]))
            feature_act = find_activation(self.activation[i])
            if feature_act is not None:
                model.add_module('act_'+str(i), feature_act)

        return model

    def forward(self, state):

        return self.module(state)


class ConvNet(nn.Module):

    def __init__(
        self,
        init_size=3,
        num_of_layers=1,
        num_of_unit=32,
        kernel_size=1,
        stride=1,
        padding=0,
        activation='relu',
        **kwargs
    ):
        super(ConvNet, self).__init__()

        if 'normalization' in kwargs.keys():
            self.normalization = kwargs['normalization']
        else:
            self.normalization = False
        if 'flatten' in kwargs.keys():
            self.flatten = kwargs['flatten']
        else:
            self.flatten = False
        if 'batch_norm' in kwargs.keys():
            self.batch_normal = kwargs['batch_norm']
        else:
            self.batch_normal = False

        self.init_size = init_size
        self.num_of_layers = num_of_layers
        self.num_of_unit = num_of_unit

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size for i in range((self.num_of_layers))]
        else:
            if len(kernel_size) != num_of_layers:
                raise RuntimeError("kernel_size is not invalid for each layer")
            else:
                self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = [stride for i in range((self.num_of_layers))]
        else:
            if len(stride) != num_of_layers:
                raise RuntimeError("stride is not invalid for each layer")
            else:
                self.stride = stride

        if isinstance(padding, int):
            self.padding = [padding for i in range((self.num_of_layers))]
        else:
            if len(padding) != num_of_layers:
                raise RuntimeError("padding is not invalid for each layer")
            else:
                self.padding = padding

        if isinstance(num_of_unit, int):
            self.num_of_unit = [num_of_unit for i in range((self.num_of_layers))]
        else:
            if len(num_of_unit) != num_of_layers:
                raise RuntimeError("padding is not invalid for each layer")
            else:
                self.num_of_unit = num_of_unit

        if isinstance(activation, str):
            self.activation = [activation for i in range(self.num_of_layers)]
        else:
            if len(activation) != num_of_layers:
                raise RuntimeError("activation is not invalid for each layer")
            else:
                self.activation = activation
        self.activation = activation

        self.module = self.build_model()

    def build_model(self):

        model = nn.Sequential()
        init_size = self.init_size
        if self.normalization:
            model.add_module('normalization', Nomramlization())
        for i in range(self.num_of_layers):

            model.add_module(
                'layer_'+str(i),
                nn.Conv2d(
                    init_size,
                    self.num_of_unit[i],
                    kernel_size=self.kernel_size[i],
                    stride=self.stride[i],
                    padding=self.padding[i]
                )
            )
            init_size = self.num_of_unit[i]
            feature_act = find_activation(self.activation[i])

            if self.batch_normal:
                model.add_module('batch_norm_'+str(i), nn.BatchNorm2d(self.num_of_unit[i]))

            model.add_module('act+' + str(i), feature_act)

        if self.flatten:
            model.add_module('flatten', Flatten())

        return model

    def forward(self, state):
        return self.module(state)


if __name__ == "__main__":
    a = ConvNet()