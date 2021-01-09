import torch
import torch.nn as nn
import numpy as np
import json


class Json_Config_Parser:

    def __init__(self, file_name):

        with open(file_name) as json_file:
            self.json_data = json.load(json_file)

    def load_parser(self):
        return self.json_data

    def load_agent_parser(self):
        data = self.json_data.get('agent')
        data['state_size'] = self.json_data['state_size']
        data['action_size'] = self.json_data['action_size']
        data['action_size_rl'] = self.json_data['action_size_rl']
        data['device'] = self.json_data['gpu_name']
        data['discount_factor'] = self.json_data['discount_factor']
        return data

    def load_local_agent_parser(self, **kwargs):
        temp = self.load_agent_parser()
        data = {}
        data['parm'] = temp
        for key, value in kwargs.items():
            data[key] = value
        return data

    def load_optimizer_parser(self):
        return self.json_data.get('optimizer')


def calculate_global_norm(agent):

    total_norm = 0

    for p in agent.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    return total_norm


def clipping_by_global_norm(agent, max_norm):
    total_norm = calculate_global_norm(agent)

    for p in agent.parameters():
        temp = max_norm/np.maximum(total_norm, max_norm)
        p.grad = p.grad*temp


def get_optimizer(parms, agent):
    if parms['name'] == 'adam':
        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            eps=1e-7)
    elif parms['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            agent.parameters(),
            lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            momentum=parms['momentum'])
    elif parms['name'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            agent.parameters(),
            lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            eps=parms['eps'])
    else:
        optimizer = None

    return optimizer


def get_optimizer_weight(parms, weight):
    if parms['name'] == 'adam':
        optimizer = torch.optim.Adam(
            weight, lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            eps=1e-7)
    elif parms['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            weight,
            lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            momentum=parms['momentum'])
    elif parms['name'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            weight,
            lr=parms['learning_rate'],
            weight_decay=parms['weight_decay'],
            eps=parms['eps'])
    else:
        optimizer = None

    return optimizer


class Exp_nn(nn.Module):
    def __init__(self):
        super(Exp_nn, self).__init__()

    def forward(self, x):
        return x.exp()


class Multiply_nn(nn.Module):
    def __init__(self, scaling):
        super(Multiply_nn, self).__init__()
        self.scaling = scaling

    def forawrd(self, x):
        return x*self.scaling


class Adding_nn(nn.Module):
    def __init__(self, bias):
        super(Adding_nn, self).__init__()
        self.bias____ = bias

    def forward(self, x):
        return x + self.bias____


def find_activation(string):
    string = string.lower()

    if string == 'relu':
        activation = torch.nn.ReLU()
    elif string == 'leaky_relu':
        activation = torch.nn.LeakyReLU()
    elif string == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif string == 'tanh':
        activation = torch.nn.Tanh()
    elif string == 'exp':
        activation = Exp_nn()
    elif string == 'linear':
        activation = None
    else:
        activation = torch.nn.ReLU()

    return activation


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()


def get_output_size(model, input_size):
    input_size[0] = 1
    a = torch.rand(input_size)
    output_size = model.forward(a).shape
    return output_size


class PidPolicy:
    def __init__(self, parm):
        self.parm = parm

    def pid_policy(self, dx, dy, yaw):
        """
        PID policy in safe situation

        Kp_lin : 종 방향 gain
        Kp_ang : 횡 방향 gain
        """
        self.dx = dx
        self.dy = dy
        self.yaw = yaw

        e_s, e_yaw = self.calculate_e()
        uv_pid = self.parm['Kp_lin'] * e_s
        uw_pid = self.parm['Kp_ang'] * e_yaw

        uv_pid = np.clip(uv_pid, self.parm['uv_min'], self.parm['uv_max'])
        uw_pid = np.clip(uw_pid, self.parm['uw_min'], self.parm['uw_max'])

        return uv_pid, uw_pid

    def calculate_e(self):
        """
        Calculate longitudinal and lateral error for PID policy
        """
        e_s = np.hypot(self.dx, self.dy) * np.cos(np.arctan2(self.dy, self.dx) - self.yaw)
        yaw_ref = np.arctan2(self.dy, self.dx)
        e_yaw_ = yaw_ref - self.yaw
        e_yaw = np.arctan2(np.sin(e_yaw_), np.cos(e_yaw_))
        return e_s, e_yaw


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class Nomramlization(nn.Module):
    def __init__(self):
        super(Nomramlization, self).__init__()

    def forward(self, x):
        if torch.is_tensor(x) is False:
            x = torch.tensor(x)
        x = x.float()

        x = x/255.
        return x


class OUNoise(object):
    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) *\
            min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
