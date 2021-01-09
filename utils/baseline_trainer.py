import torch
import gym
import numpy as np
import datetime
import cv2

from collections import deque
from utils.utils import Json_Config_Parser
from mlagents.envs import UnityEnvironment


"""
env_name : the name of environment. it can be generated from unity, gym, atari, mujoco

state_size : the input shape of network

action_size : the output shape of network

agent : information about network configuration

run_step : how many step you run!!

start_step: to buffer replay memeory, initially run without training

show_episode

size_replay_memory

batch_size

learning_freq

"""


class Policy:

    def __init__(self, file_name):
        self.parser = Json_Config_Parser(file_name)
        self.parm = self.parser.load_parser()
        self.agent_parms = self.parser.load_agent_parser()
        self.optimizer_parms = self.parser.load_optimizer_parser()
        self.agent = torch.nn.Sequential()

        self.target_agent = torch.nn.Sequential()
        self.target_agent.load_state_dict(self.agent.state_dict())

        if 'unity_env' in self.parm.keys():
            if self.parm['unity_env'] == 'True':
                self.u_model = True
            else:
                self.u_model = False
        else:
            self.u_model = False
        if 'conservative_mode' in self.parm.keys():
            self.c_mode = self.parm['conservative_mode'] == 'True'
            self.tau = self.parm['tau']
        else:
            self.c_mode = False
            self.update_step = self.parm['update_step']

        if self.u_model:
            name = './env/MobileRobot_SingleBrain_following_old/Windows/MobileRobot_SingleBrain_following_old'
            # name = './env/MobileRobot_SingleBrain_following/Windows/MobileRobot_SingleBrain_following'
            self.env = UnityEnvironment(file_name=name)
            self.default_brain = self.env.brain_names[0]
            self.brain = self.env.brains[self.default_brain]
        else:
            self.env = gym.make(self.parm['env_name'])

        self.obs_set = deque(maxlen=self.parm['state_size'][0])
        self.replay_memory = deque(maxlen=int(self.parm['size_replay_memory']))
        self.optimizer = None
        self.discount_factor = self.parm['discount_factor']

        if 'linear_decay' in self.optimizer_parms.keys():
            self.decaying_mode = self.optimizer_parms['linear_decay'] == 'True'
        else:
            self.decaying_mode = False
        if 'learning_rate' in self.optimizer_parms.keys():
            self.lr = self.optimizer_parms['learning_rate']
        self.state_size = self.parm['state_size']
        self.following_safety_dist = self.parm['following_safety_dist']
        self.action_size = self.parm['action_size']
        self.action_size_rl = self.parm['action_size_rl']

        self.gpu = torch.device(self.parm['gpu_name'])
        self.cpu = torch.device('cpu')

        self.Exploration_method = 'e_greedy'

        if 'Exploration_method' in self.parm.keys():
            self.Exploration_method = self.parm['Exploration_method']

        self.epsilon_min = .1
        if 'epsilon_min' in self.parm.keys():
            self.epsilon_min = self.parm['epsilon_min']
        self.batch_size = self.parm['batch_size']

        self.run_step = self.parm['run_step']
        self.start_step = self.parm['start_step']
        self.show_episode = self.parm['show_episode']

        self.save_path = self.parm['save_path']
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.save_path = self.save_path + self.parm['env_name'] + '_' + date_time + '.pth'
        if 'reward_scaling' in self.parm.keys():
            self.reward_scaling = self.parm['reward_scaling']

        name = self.parm['tensorboard_path'] + \
            self.parm['env_name']+'_' + date_time + '_' + \
            'reward_scaling_'+str(self.reward_scaling)

        if 'fixed_temperature' in self.parm.keys():
            if self.parm['fixed_temperature'] == 'True':
                name += '_Fixed'
        # self.writer = SummaryWriter(name + '/')

        self.inference_mode = self.parm['inference_mode'] == 'True'

        self.reward_scaling = 1

        if len(self.state_size) >= 3:
            self.mode = 'Image'
        else:
            self.mode = 'Vector'

    def reset(self):

        for i in range(self.state_size[0]):
            self.obs_set.append(np.zeros([i for i in self.state_size[1:]]))

    def generate_optimizer(self):
        pass

    def lr_scheduler(self, step):
        p = step / self.run_step / 10
        self.lr = self.optimizer_parms['learning_rate'] * (1 - p)

        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def preprocess_state(self, obs):

        state = np.zeros(self.parm['state_size'])

        if self.mode == 'Image':
            obs = cv2.resize(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (self.parm['state_size'][1], self.parm['state_size'][2]))
        self.obs_set.append(obs)
        print(self.parm['state_size'][0])
        for i in range(self.parm['state_size'][0]):
            state[i] = self.obs_set[i]

        if self.agent.mode == 'Image':
            state = np.uint8(state)
        return state

    def append_memory(self, sarsd):
        self.replay_memory.append(sarsd)

    def get_action(self, state):

        pass

    def target_network_update(self, step):
        if self.c_mode:
            with torch.no_grad():
                for t_pa, pa in zip(self.target_agent.parameters(), self.agent.parameters()):
                    temp = self.tau * pa + (1-self.tau) * t_pa
                    t_pa.copy_(temp)
        elif step % self.update_step == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

    def train(self):
        pass

    def run(self):

        losses = []
        rewards = []
        norms = []
        step = 0
        episode = 0
        mode = self.parm['inference_mode'] != 'True'
        if self.u_model:
            env_info = self.env.reset(train_mode=mode)[self.default_brain]

        breakvalue = 1
        while breakvalue:

            episode_loss = []
            episode_norm = []
            episode_reward = 0

            if self.u_model:
                ob = 255 * np.array(env_info.visual_observations[0])[0]
            else:
                ob = self.env.reset()
            self.reset()
            state = self.preprocess_state(ob)
            done = False

            while done is False:
                a = self.get_action(state)
                if self.u_model:
                    env_info = self.env.step([a])[self.default_brain]
                    obs = 255 * np.array(env_info.visual_observations[0])[0]
                    reward = env_info.rewards[0]
                    done = env_info.local_done[0]
                else:
                    obs, reward, done, _ = self.env.step(a)

                step += 1
                next_state = self.preprocess_state(obs)

                self.append_memory((state, a, reward*self.reward_scaling, next_state, done))
                if step >= self.start_step and self.inference_mode is False:
                    if self.decaying_mode:
                        self.lr_scheduler(step)
                    if step & self.parm['learning_freq'] == 0:
                        loss, norm = self.train()
                        episode_loss.append(loss.to(self.cpu).detach().numpy())
                        episode_norm.append(norm)

                state = next_state
                episode_reward += reward
                if self.parm['render_mode'] == 'True' and self.u_model is False:
                    self.env.render()
                if step > self.start_step:
                    self.target_network_update(step)
                if done:
                    episode_loss = np.array(episode_loss).mean()
                    episode_norm = np.array(episode_norm).mean()

                    self.writer.add_scalar('Loss', episode_loss, step)
                    self.writer.add_scalar('Reward', episode_reward, step)
                    self.writer.add_scalar('Norm', episode_norm, step)

                    losses.append(episode_loss)
                    rewards.append(episode_reward)
                    norms.append(episode_norm)
                    episode += 1

                    if (episode + 1) % self.show_episode == 0 or self.inference_mode:
                        losses = np.array(losses).mean()
                        rewards_ = np.array(rewards).mean()
                        norms = np.array(norms).mean()
                        print(
                            """Episode : {:4d} // Step : {:5d} // Loss: {:3f} //
                               Reward : {:3f} // Norm : {:3f}""".format(
                                episode + 1, step, losses, rewards_, norms))

                        losses = []
                        rewards = []
                        norms = []

                    if step > self.start_step and (episode + 1) % 10 == 0 and\
                       self.inference_mode is False:
                        torch.save(self.agent.state_dict(), self.save_path)
