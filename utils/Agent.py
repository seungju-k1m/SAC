import torch
import torch.nn as nn
from utils.baseline_agent import Base_Agent, MLP, ConvNet
from utils.utils import get_output_size


class SAC_Agent(Base_Agent):
    """
    SAC Agent는 sac algorithm을 지원하는 agent를 위한 class이다.

    하나의 actor와 두 개의 critic을 포함하고 있으며, forwarding methods를 통해 output을 반환한다.

    또한 sac에 요구되는 objective function을 계산하고 이를 반환하여, trainer에서 update가 이루어 질 수 있도록 지원한다.

    """

    def __init__(self, parms):
        super(SAC_Agent, self).__init__()
        self.parms = parms
        self.batch_norm = False
        if self.parms['network'] == 'MLP':
            self.init_size = self.parms['state_size'][0] * self.parms['state_size'][-1]
            self.critic_init_size = self.init_size + self.parms['action_size_rl']
            self.state_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][-1]]
            self.mode = 'MLP'
        else:
            self.init_size = self.parms['state_size'][0]
            self.normalization = self.parms['normalization'] == 'True'
            if 'batch_norm' in self.parms.keys():
                self.batch_norm = self.parms['batch_norm'] == 'True'

            self.kernel_size = list(self.parms['kernel_size'])
            self.padding = list(self.parms['padding'])
            self.stride = list(self.parms['stride'])

            self.mode = 'Conv'

        self.action_size = self.parms['action_size_rl']
        self.device = torch.device(self.parms['device'])

        self.actor_parm = self.parms['actor']
        self.a_n_layers = self.actor_parm['num_of_layers']
        self.a_n_unit = self.actor_parm['filter_size']
        self.a_act = self.actor_parm['activation']
        self.a_l_act = self.actor_parm['actor_activation']
        self.a_bn = self.actor_parm['batch_norm'] == 'True'

        self.critic_parm = self.parms['critic']
        self.c_n_layers = self.critic_parm['num_of_layers']
        self.c_act = self.critic_parm['activation']
        self.c_n_unit = self.critic_parm['filter_size']
        self.c_bn = self.critic_parm['batch_norm'] == 'True'

        self.input_size = [-1]
        if self.mode == 'MLP':
            self.input_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][1]]
            self.critic_input_size = [-1, self.parms['state_size'][0] * self.parms['state_size'][1]
                                      + self.parms['action_size_rl']]
        elif self.mode == 'Conv':
            for i in self.parms['state_size']:
                self.input_size.append(i)

        self.alpha = torch.ones((1))

        self.build_model()

    def build_model(self):
        """
        Configuration에 따라 model을 build 한다.
        """
        self.actor = nn.Sequential()
        if self.mode == 'MLP':

            self.actor_Feature = MLP(init_size=self.init_size,
                                     num_of_layers=self.a_n_layers,
                                     num_of_unit=self.a_n_unit,
                                     activation=self.a_act,
                                     batch_norm=self.a_bn).module

            shape = get_output_size(self.actor_Feature.eval(), self.input_size.copy())
            self.actor_Feature.to(self.device)

            # self.actor = MLP(init_size=shape[-1],
            #                          num_of_layers=1,
            #                          num_of_unit=self.parms['action_size'],
            #                          activation=self.a_l_act,
            #                    batch_norm=self.a_bn).module.to(self.device)
            self.actor = nn.Linear(
                in_features=shape[-1],
                out_features=self.action_size).to(self.device)

            self.policy = nn.Linear(
                in_features=shape[-1],
                out_features=self.action_size).to(self.device)
            # self.policy = nn.Linear(in_features=shape[-1], out_features=1).to(self.device)

            self.critic_ = MLP(init_size=self.critic_init_size,
                               num_of_layers=self.c_n_layers,
                               num_of_unit=self.c_n_unit,
                               activation=self.c_act,
                               batch_norm=self.c_bn).module

            self.critic_2 = MLP(init_size=self.critic_init_size,
                                num_of_layers=self.c_n_layers,
                                num_of_unit=self.c_n_unit,
                                activation=self.c_act,
                                batch_norm=self.c_bn).module

            # self.Temperature = nn.Sequential()
            # self.Temperature.add_module('fc1', torch.nn.Linear(1,1))
            self.Temperature = torch.zeros((1), requires_grad=True, device="cuda")
            # def temperature_init(m):
            #     if isinstance(m, nn.Linear):
            #         torch.nn.init.uniform(m.weight, -1.3, -1.0)
            #         torch.nn.init.uniform(m.bias, -0.1, -0.01)
            # self.Temperature.apply(temperature_init)

        elif self.mode == 'Conv':
            self.Feature = ConvNet(init_size=self.init_size,
                                   num_of_layers=self.num_of_layers,
                                   num_of_unit=self.num_of_unit,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   flatten=True,
                                   batch_norm=self.batch_norm
                                   ).module

        if self.c_bn:
            self.critic = nn.Sequential(*list(self.critic_.children())[:-2]).to(self.device)
            self.critic2 = nn.Sequential(*list(self.critic_2.children())[:-2]).to(self.device)
        else:
            self.critic = nn.Sequential(*list(self.critic_.children())[:-1]).to(self.device)
            self.critic2 = nn.Sequential(*list(self.critic_2.children())[:-1]).to(self.device)

        # self.critic,2 , state + action >> q(s,a)
        # self.Temperature.to(self.device)

        # self.Temperature > state + action >> alpha/(ReLu) > 0

    def forward(self, state):

        # state를 tensor형태로 변환하는 과정이다.
        if torch.is_tensor(state) is False:
            state = torch.tensor(state).to(self.device).float()
        state = state.view(self.state_size).to(self.device).float()

        # mean, logstd를 구하기 위해서 actor Feature를 계산한다.
        actor_feature = self.actor_Feature(state)

        # 구해진 actor feature를 actor, policy에 넣어준다.
        mean = self.actor(actor_feature)
        log_std = self.policy(actor_feature)
        # log_std = torch.clamp(self.policy(actor_feature),-20,34)

        # mean, log_std를 이용하여, gaussian sampling을 한다.
        # reparameterization trick에 대해서 공부하면 gaussian sampling을 더 잘 이해할 수 있다.
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        log_prob -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        # state와 action을 critic에 넣어주어, state-action value를 구한다.
        concat = torch.cat((state, action), dim=1)
        critic = self.critic(concat)
        critic2 = self.critic2(concat)
        return action, log_prob, (critic, critic2), entropy

    def loss(self,
             states,
             targets,
             actions,
             alpha=0):

        # 훈련하기전 훈련 모드로 전환한다.
        self.actor_Feature.train()
        self.actor.train()
        self.critic.train()
        self.critic2.train()
        # self.Temperature.train()

        # state를 tensor로 변환해 준다.
        if torch.is_tensor(states) is False:
            states = torch.tensor(states).to(self.device).float()
        states = states.view(self.state_size).to(self.device).float()

        # forward를 통해, 훈련에 필요한 값들을 구한다.
        action, log_prob, critics, entropy = self.forward(states)
        target1, target2 = targets
        state_action = torch.cat((states, actions), dim=1)

        critic1 = self.critic(state_action.detach())
        critic2 = self.critic2(state_action.detach())

        # critic을 위한 objective function을 계산한다.
        loss_critic1 = torch.mean((critic1-target1).pow(2))/2
        loss_critic2 = torch.mean((critic2-target2).pow(2))/2

        # forward를 통해 구한 critic은 actor를 update할 때 필요로한다.
        critic1_p, critic2_p = critics

        critic_p = torch.min(critic1_p, critic2_p)
        temperature = self.Temperature
        loss_policy = torch.mean(temperature * log_prob - critic_p)
        # loss_policy = torch.mean(-temperature_ * entropy-critic_p.clone())

        loss_temp = torch.mean(temperature.exp() * (-log_prob+self.action_size))
        # loss_temp = torch.mean(temperature*(entropy_+self.action_size))
        self.alpha = temperature.exp()
        return loss_critic1, loss_critic2, loss_policy, loss_temp
