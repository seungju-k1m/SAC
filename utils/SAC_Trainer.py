import torch
import random
import numpy as np
import cv2
import gym

from utils.baseline_trainer import Policy
from utils.Agent import SAC_Agent
from utils.utils import get_optimizer, get_optimizer_weight, calculate_global_norm
from torchsummary import summary
from collections import deque
from utils.utils import PidPolicy


class action_space:

    def __init__(self, action_size):
        self.shape = [action_size]
        self.low = -1
        self.high = 1


class SAC_Trainer(Policy):
    """
    SAC_Trainer는 알고리즘 전체 과정을 제어하는 역할을 수행한다.

    그 역할을 다음과 같이 정렬하면

        1. set the hyper parameter from configuration file.
        2. sample from the environment.
        3. training
        4. logging
        5. saving
        6. uploading
        7. evaluating
    """

    def __init__(self, file_name):
        super(SAC_Trainer, self).__init__(file_name)
        if 'fixed_temperature' in self.parm.keys():
            self.fixed_temperature = self.parm['fixed_temperature'] == 'True'
            if 'temperature_parameter' in self.parm.keys():
                self.temperature_value = self.parm['temperature_parameter']
        else:
            self.fixed_temperature = False
        torch.manual_seed(self.parm['seed'])
        np.random.seed(self.parm['seed'])
        # self.env.seed(self.parm['seed'])

        self.agent = SAC_Agent(self.agent_parms).train()
        if self.parm['load_model'] == "True":
            self.agent.load_state_dict(
                torch.load(self.parm['load_path'], map_location=self.parm['gpu_name']))
            print("\n ---------- Model {} is Loaded! --------- \n".format(self.parm['load_path']))
        self.target_agent = SAC_Agent(self.agent_parms)

        self.actor_feature, self.actor, self.policy, self.critic1, self.critic2, self.temperature =\
            self.agent.actor_Feature, self.agent.actor, self.agent.policy,\
            self.agent.critic, self.agent.critic2, self.agent.Temperature

        self.target_critic1, self.target_critic2 =\
            self.target_agent.critic, self.target_agent.critic2

        self.target_critic1.load_state_dict(self.critic1.state_dict(), strict=False)
        self.target_critic2.load_state_dict(self.critic2.state_dict(), strict=False)
        self.actor_optimizer, self.critic_optimizer1,\
            self.critic_optimizer2, self.temperature_optimizer = self.generate_optimizer()

        self.pid = PidPolicy(self.parm)
        self._reset_num = 0
        self._saved_num = 0
        self.dx = []
        self.dy = []
        self.yaw = []
        self.uv = []
        self.uw = []

        if self.u_model is False:
            self.eval_env = gym.make(self.parm['env_name'])
            self.eval_obs_set = []

        if len(self.parm['state_size']) < 3:
            zz = np.zeros((self.parm['state_size'][0] * self.parm['state_size'][1]))
            zz = np.expand_dims(zz, 0)
            temp = tuple([self.parm['state_size'][0] * self.parm['state_size'][1]])

            print('Actor')
            summary(self.actor_feature.to(self.gpu), temp)

            zz = np.zeros((
                self.parm['state_size'][0] * self.parm['state_size'][1]+self.action_size_rl))
            zz = np.expand_dims(zz, 0)
            temp = tuple(
                [self.parm['state_size'][0] * self.parm['state_size'][1]+self.action_size_rl])
            print('Critic')
            summary(self.critic1.to(self.gpu), temp)

            # summary(self.temperature.to(self.gpu),tuple([1]))

        else:
            zz = np.zeros((self.parm['state_size']))
            zz = np.expand_dims(zz, 0)
            summary(self.agent.to(self.gpu), tuple(self.parm['state_size']))
            # self.writer.add_graph(self.agent, torch.tensor(zz).float().to(self.gpu))

        self.flatten_state_size = [-1, self.state_size[0] * self.state_size[1]]
        self.brain_name = self.env.brain_names[0]

        # for name in self.env.brain_names:
        #     self.brain_name.append(name)

        zeta = self.agent_parms['temperature']

        self.t_scaling = zeta['temperature_scaling']
        self.t_offset = zeta['temperature_offset']
        self.gradient_steps = self.parm['gradient_steps']
        self.agent_num = self.parm['multi_agent']
        if 'input_normalization' in self.parm.keys():
            self.input_norm = self.parm['input_normalization'] == 'True'
        else:
            self.input_norm = False
        self.obs_set = []
        for i in range(self.agent_num):
            self.obs_set.append(deque(maxlen=self.parm['state_size'][0]))

        self.step0_f_list = np.zeros((self.agent_num-1, 1))
        self.dist_th = self.parm['following_safety_dist']

    def reset(self):
        """
            obs_set을 초기화한다.
        """
        for j in range(self.agent_num):
            for i in range(self.state_size[0]):
                self.obs_set[j].append(np.zeros([k for k in self.state_size[1:]]))

    def preprocess_state(self, obs, id):
        """
        observation을 state로 변환한다.
        """
        state = np.zeros(self.parm['state_size'])
        if self.mode == 'Image':
            obs = cv2.resize(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
        self.obs_set[id].append(obs)

        for i in range(self.parm['state_size'][0]):
            state[i] = self.obs_set[id][i]

        if self.agent.mode == 'Image':
            state = np.uint8(state)
        return state

    def generate_optimizer(self):

        """
        optimizer를 반환한다.

        이때 actor, critic, temperature를 따로 반환하는 이유는

        순차적으로 update되기 때문이다.
        """
        weights = list(self.actor.parameters())+list(self.policy.parameters())
        weights = weights + list(self.actor_feature.parameters())
        actor_optimizer = get_optimizer_weight(self.optimizer_parms['actor'], weights)
        critic_optimizer1 = get_optimizer(self.optimizer_parms['critic'], self.critic1)
        critic_optimizer2 = get_optimizer(self.optimizer_parms['critic'], self.critic2)
        if self.fixed_temperature is False:
            temperature_optimizer = get_optimizer_weight(
                self.optimizer_parms['temperature'], [self.temperature])
        else:
            temperature_optimizer = None

        return actor_optimizer, critic_optimizer1, critic_optimizer2, temperature_optimizer

    def get_action(self, state):
        """
        get_action method는 RL algorithm에 해당하는 action을 반환한다.
        """

        # state를 네트워크에 입력할 수 있도록 전처리를 해준다.
        if ~torch.is_tensor(state):
            state = torch.tensor(state).to(self.gpu).float()
        size = [-1]
        for i in self.state_size:
            size.append(i)
        state = state.view(size)

        # 이를 기반으로 agent forward를 통해, action을 구한다.
        action, log_std, critics, _ = self.agent.eval().forward(state)

        # action을 tensor에서 numpy.ndarray로 변환
        action = action.to(self.cpu).detach().numpy()

        # 마지막에 0을 추가하여, 환경에게 RL approach라고 알린다.
        action = np.append(action, 0)
        return action

    def get_action_deterministic(self, state, state_f, step_f):
        """
        get_action_deterministic은 RL algorithm뿐만 아니라, safety, pid algorithm에 해당하는
        행동을 반환한다.

        action 3번째 term : (0 -> rl policy 사용시), (1 -> pid policy 사용시)
        """
        # state를 전처리한다.
        if ~torch.is_tensor(state):
            state = torch.tensor(state).to(self.gpu).float()
            state_f = torch.tensor(state_f).float()
        size = [-1, self.state_size[0]*self.state_size[1]]
        state = state.view(size)

        obs = state[0, 6:].cpu().numpy()
        id = int(state_f.item())
        yaw = -state[0, 2].item()

        dx = state[0, 0].item()
        dy = state[0, 1].item()

        # goal 까지의 거리
        dist_to_goal = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

        # print("------------------------")
        # print("id : ", id)
        # print("dist_to_goal : {:.2f} [m]".format(dist_to_goal))

        obs_dist = obs * self.parm['lidar_roi_dist']

        # 장애물과의 거리(np.min(obs_dist)가 safety distance (r_safe) 보다 크고 goal 까지의 거리보다 클 경우 "pid mode"
        if np.min(obs_dist) > self.parm['r_safe'] or np.min(obs_dist) > dist_to_goal:
            # print("--- pid mode ---")
            uv_pid, uw_pid = self.pid.pid_policy(dx, dy, yaw)
            action = np.array([[uv_pid, -uw_pid, 1]])

        # 장애물과의 거리(np.min(obs_dist)가 risk distance (r_risk) 보다 작을 경우 "safety rl mode"
        elif np.min(obs_dist) <= self.parm['r_risk']:
            # print("--- safety rl mode ---")
            state_obs = self.safety_policy(state)
            actor_feature = self.agent.actor_Feature(state_obs)
            action = torch.tanh(self.actor(actor_feature))
            action = action.to(self.cpu).detach().numpy()
            action = np.append(action, 0)
            action = np.reshape(action, (1, 3))

        # 그 외의 경우 "rl mode"
        else:
            # print("--- rl mode ---")
            actor_feature = self.agent.actor_Feature(state)
            action = torch.tanh(self.actor(actor_feature))
            action = action.to(self.cpu).detach().numpy()
            action = np.append(action, 0)
            action = np.reshape(action, (1, 3))

        if step_f == 0 and id == 0:
            action = np.array([[0, 0, 1]])

        '''
        - id 가 0 이 아닌 agent (리드로봇이 아닌 로봇들) 이 각각의 goal 까지
        dist_th (threshold 거리) 안으로 들어오면 step0_f_list 을 1로 업데이트 시킴
        (리드로봇 제외한 모든 로봇이 step0_f_list 이 1로 되면
        following step (step_f) = 1 로 넘어감)

        - 충돌 방지를 위해 안전거리 안으로 들어오면 action 을 0으로 넣어줘서 움직임 없앰.
        (id 가 0 인 리드로봇은 goal 로 들어가기 때문에 해당사항 없음
        id 가 0 이 아닌 로봇들은 앞 로봇을 goal 로 삼기 때문에 충돌 방지 필요)
        '''
        if id != 0 and dist_to_goal < self.dist_th:
            self.step0_f_list[id-1] = 1
            action = np.array([[0, 0, 1]])
        return action

    def safety_policy(self, state):
        # p_scale 만큼 lidar observation 거리를 짧게 만들어줘 더 위험한 상황으로 인식하고 대처하도록 하기위함
        state[0, 6:-1] = state[0, 6:-1] / self.parm['p_scale']
        return state

    def target_network_update(self, step):
        """
        target network를 update한다.
        """
        if self.c_mode:
            k = 0
            with torch.no_grad():
                for t_pa1, t_pa2, pa1, pa2 in zip(
                    self.target_critic1.parameters(),
                    self.target_critic2.parameters(),
                    self.critic1.parameters(),
                    self.critic2.parameters()
                ):

                    if self.input_norm and k < 2:
                        temp1 = self.tau * pa1 + (1 - self.tau) * t_pa1
                        temp2 = self.tau * pa2 + (1 - self.tau) * t_pa2
                    else:
                        temp1 = self.tau * pa1 + (1 - self.tau) * t_pa1
                        temp2 = self.tau * pa2 + (1 - self.tau) * t_pa2
                    t_pa1.copy_(temp1)
                    t_pa2.copy_(temp2)
                    k += 1

    def preprocess_state_eval(self, obs):

        state = np.zeros(self.parm['state_size'])

        if self.mode == 'Image':
            obs = cv2.resize(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
        self.obs_set.append(obs)

        for i in range(self.parm['state_size'][0]):
            state[i] = self.obs_set[i]

        if self.agent.mode == 'Image':
            state = np.uint8(state)
        return state

    def zero_grad(self):
        """
        각 optimizer에 zero_grad를 반환한다.
        """
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        self.actor_optimizer.zero_grad()
        if self.fixed_temperature is False:
            self.temperature_optimizer.zero_grad()

    def train(self):
        # replay memory로부터 sample은 얻는다.
        mini_batch = random.sample(self.replay_memory, self.batch_size)

        # agent를 train mode로 설정
        self.agent = self.agent.train()

        # s, a, r, s_, d를 전처리 한다.
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        next_actions = []

        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append([mini_batch[i][1]])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        actions = torch.tensor(actions).view(
            self.batch_size, self.action_size_rl).to(self.gpu).float()
        next_states = torch.tensor(next_states).view(self.flatten_state_size).float().to(self.gpu)
        states = torch.tensor(states).view(self.flatten_state_size).float().to(self.gpu)
        next_actions, log_prob, _, entropy = self.agent.eval().forward(next_states)
        next_state_action = torch.cat((next_states, next_actions), im=1)

        target1 = self.target_critic1.eval()(next_state_action)
        target2 = self.target_critic2.eval()(next_state_action)
        temperature = self.temperature.detach().view((-1, 1))

        # 이를 기반으로 target value를 구한다.
        if self.fixed_temperature:
            alpha = self.temperature_value
        else:
            alpha = temperature.exp()

        for i in range(self.batch_size):
            if dones[i]:
                target1[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
                target2[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
            else:
                temp = -alpha * log_prob[i]
                # temp = alpha * entropy[i]
                target1[i] = rewards[i] + self.discount_factor * (target1[i]+temp)
                target2[i] = rewards[i] + self.discount_factor * (target2[i]+temp)

        # 구한 target값을 이용하여, objective function을 구한다.
        # agent class에 objective function에 대해서 설명되어있다.
        if self.fixed_temperature:
            loss_critic1, loss_critic2, loss_policy, loss_temp =\
                self.agent.loss(
                    states.detach(),
                    (target1.detach(), target2.detach()),
                    actions.detach(),
                    alpha=alpha)

        else:
            loss_critic1, loss_critic2, loss_policy, loss_temp =\
                 self.agent.loss(
                     states.detach(),
                     (target1.detach_(), target2.detach()),
                     actions.detach())

        # zero grad를 통해 update를 할 수 있도록 준비한다.
        self.zero_grad()
        loss_policy.backward()

        # actor를 update하는 과정에서, critic에도 grad가 생기게 된다.
        # 이를 zero grad로 반환하여 없애준다.
        self.critic1.zero_grad()
        self.critic2.zero_grad()

        # critic을 update한다.
        loss_critic1.backward()
        loss_critic2.backward()
        if self.fixed_temperature:
            pass
        else:
            loss_temp.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        self.actor_optimizer.step()
        if self.fixed_temperature is False:
            self.temperature_optimizer.step()

        # logging을 하기 위해서 gradient의 크기를 구한다.
        z = calculate_global_norm(self.actor_feature)
        z += calculate_global_norm(self.critic2)
        z += calculate_global_norm(self.critic1)
        z += calculate_global_norm(self.policy)
        z += calculate_global_norm(self.actor)

        return loss_policy, (loss_critic1+loss_critic2)/2,\
            loss_temp, z, entropy.mean().detach().to(self.cpu).numpy()

    def run(self):
        losses = []
        rewards = []
        norms = []
        step = 0
        episode = 0
        mode = self.parm['inference_mode'] != 'True'

        '''
        "following mode" : 1 -> following mode on (학습 된 모델을 이용하여 following 수행 시 1로 설정)
        "following mode" : 0 -> following mode off (학습 시 0으로 설정)
        "distance2target" : unity 상에서 goal 에 도달했다고 판단하는 거리 기준
        "distance2lead" : unity 상에서 lead 로봇에 도달했다고 판단하는 거리 기준
        '''
        env_config = {"followingMode": 1, "distance2target": 1.5, "distance2lead": 2.5}

        breakvalue = 1
        while breakvalue:
            step_f = 0
            self.step0_f_list = np.zeros((self.agent_num-1, 1))
            if step == 0:
                env_info = self.env.reset(train_mode=mode, config=env_config)[self.brain_name]
            else:
                env_info = self.env.reset(train_mode=mode)[self.brain_name]

            episode_loss_p = []
            episode_loss_c = []
            episode_loss_t = []
            episode_norm = []
            episode_entropy = []
            episode_reward = 0

            self.reset()

            state = []
            a = []

            for b in range(self.agent_num):
                print(np.array(env_info.vector_observations).shape)
                ob = np.array(env_info.vector_observations)[b]
                ob_s = ob[:self.parm['state_size'][1]]
                state_f = ob[self.parm['state_size'][1]:]
                temp = self.preprocess_state(ob_s, b)
                state.append(temp)

                if mode:
                    a.append(self.get_action(state[b]))
                else:
                    a.append(self.get_action_deterministic(state[b], state_f, step_f))
                # a.append(self.get_action(state[b], b))

                # a[self.brain_name[b]] = self.get_action(state[b])

            done_ = False
            dones = [False for i in range(self.agent_num)]

            while done_ is False:
                next_state = []
                next_state_f = []

                if self.u_model:
                    env_info = self.env.step(a)[self.brain_name]
                    for b in range(self.agent_num):
                        obs = np.array(env_info.vector_observations)[b]
                        obs_s = obs[:self.parm['state_size'][1]]
                        obs_f = obs[self.parm['state_size'][1]:]

                        reward = env_info.rewards[b]

                        done = env_info.local_done[b]
                        temp = self.preprocess_state(obs_s, b)
                        next_state_ = temp

                        if dones[b] is False:
                            episode_reward += reward
                            self.append_memory((state[b], a[b], reward * self.reward_scaling,
                                                next_state_, done, 1))

                        if done:
                            dones[b] = done

                        next_state.append(next_state_)
                        next_state_f.append(obs_f)

                step += 1
                done_temp = np.array(dones, dtype=np.float32)
                # print("done_temp : ", done_temp)
                if (np.sum(done_temp) == self.agent_num):
                    done_ = True
                # if (step+1)%1000 == 0:
                #     self.eval(step)

                if step >= self.start_step and self.inference_mode is False:
                    if self.decaying_mode:
                        self.lr_scheduler(step)
                    if step & self.parm['learning_freq'] == 0:
                        for zzz in range(self.gradient_steps):
                            loss_p, loss_c, loss_t, norm, entropy = self.train()
                            episode_loss_p.append(loss_p.to(self.cpu).detach().numpy())
                            episode_loss_c.append(loss_c.to(self.cpu).detach().numpy())
                            episode_loss_t.append(loss_t.to(self.cpu).detach().numpy())
                            episode_norm.append(norm)
                            episode_entropy.append(entropy)

                state = next_state
                state_f = next_state_f
                # print("step_f : ", step_f)
                for b in range(self.agent_num):
                    obs = np.array(env_info.vector_observations)[b]
                    state_f = obs[self.parm['state_size'][1]:]
                    if mode:
                        a[b] = self.get_action(state[b])
                    else:
                        a[b] = self.get_action_deterministic(state[b], state_f, step_f)

                '''
                following step (step_f) = 0 : 리드로봇 주변으로 집합
                following step (step_f) = 1 : 리드로봇이 goal 로 이동, 나머지 로봇 following 수행
                '''
                # 리드로봇을 제외한 모든 로봇이 "step0_f_list" 이 1로 채워지면 다음 step (step_f = 1) 로 변경
                if np.sum(self.step0_f_list) == self.parm['multi_agent']-1:
                    step_f = 1

                if self.parm['render_mode'] == 'True' and self.u_model is False:
                    self.env.render()
                if step > self.start_step:
                    self.target_network_update(step)

                if done_:
                    self._reset_num += 1

                    episode_loss_p = np.array(episode_loss_p).mean()
                    episode_loss_c = np.array(episode_loss_c).mean()
                    episode_loss_t = np.array(episode_loss_t).mean()
                    episode_norm = np.array(episode_norm).mean()
                    episode_entropy = np.array(episode_entropy).mean()

                    if self.fixed_temperature:
                        alpha = self.temperature_value
                    else:
                        alpha = self.agent.alpha.to(self.cpu).detach().numpy().mean()

                    losses.append(episode_loss_t + episode_loss_p + episode_loss_c)
                    rewards.append(episode_reward)
                    norms.append(episode_norm)
                    episode += 1

                    if (episode + 1) % self.show_episode == 0 or self.inference_mode:
                        losses = np.array(losses).mean()
                        rewards_ = np.array(rewards).mean()
                        norms = np.array(norms).mean()
                        if self.fixed_temperature:
                            alpha = self.temperature_value
                        else:
                            alpha = self.agent.alpha.to(self.cpu).detach().numpy().mean()
                        print(
                            """
                            Episode : {:4d} // Step : {:5d} // Loss: {:3f} // Reward : {:3f} //
                            Norm : {:3f} // alpha: {:3f}
                            """.format(
                                episode + 1, step, losses, rewards_, norms, alpha))

                        losses = []
                        rewards = []
                        norms = []
                    if step > self.start_step and (episode + 1) % 10 == 0 and\
                       self.inference_mode is False:
                        torch.save(self.agent.state_dict(), self.save_path)
                        print("---- Model is saved ----")
