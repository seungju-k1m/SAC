import os
import gc
import copy
import psutil
import time
import torch
import datetime
import numpy as np
from baseline.baseTrainer import ONPolicy
from PPO.Agent import ppoAgent
from baseline.utils import getOptim, PidPolicy
from collections import deque
from PPO.wrapper import preprocessBatch, preprocessState
# from mlagents_envs.base_env import ActionTuple


class PPOOnPolicyTrainer(ONPolicy):
    """
    PPOOnPolicyTrainer는 알고리즘 전체 과정을 제어하는 역할을 수행한다.

    그 역할을 다음과 같이 정렬하면

        1. set the hyper parameter from configuration file.
        2. sample from the environment
        3. training
        4. logging
        5. saving
        6. uploading
        7. evaluating
    """

    def __init__(self, cfg):
        """
        configuration에 따라 actor, critic, optimizer등을 반환한다.
        """
        super(PPOOnPolicyTrainer, self).__init__(cfg)

        torch.set_default_dtype(torch.float64)
        torch.backends.cudnn.benchmark = True

        if self.device != "cpu":
            self.device = torch.device(self.device)

        else:
            self.device = torch.device("cpu")

        self.entropyCoeff = self.data['entropyCoeff']
        self.epsilon = self.data['epsilon']
        self.labmda = self.data['lambda']
        initLogStd = torch.tensor(self.data['initLogStd']).to(self.device)
        finLogStd = torch.tensor(self.data['finLogStd']).to(self.device)
        annealingStep = self.data['annealingStep']
        self.LSTMName = self.data['LSTMName']
        self.agent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMName=self.LSTMName)
        self.agent.to(self.device)
        if self.lPath != "None":
            self.agent.load_state_dict(
                torch.load(self.lPath, map_location=self.device)
            )
            self.agent.loadParameters()
        self.oldAgent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMName=self.LSTMName)
        self.oldAgent.to(self.device)
        self.oldAgent.update(self.agent)
        self.copyAgent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMName=self.LSTMName)
        self.copyAgent.to(self.device)
        self.copyAgent.update(self.agent)

        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.sPath += name + '_' + str(time)+'.pth'

        self.sSize = self.aData['sSize']
        self.updateStep = self.data['K1']
        self.genOptim()

        self.div = self.data['div']

        self.replayMemory = [deque(maxlen=self.updateStep) for i in range(self.div)]
        self.epoch = self.data['epoch']
        self.updateOldP = self.data['updateOldP']
        self.Number_Episode = 0
        self.Number_Sucess = 0

        self.ReplayMemory_Trajectory = deque(maxlen=1000000)
        self.K1 = self.data['K1']
        self.K2 = self.data['K2']
        self.RecordScore = self.data['RecordScore']

        self.pid = PidPolicy(self.data)
        self._reset_num = 0
        self._saved_num = 0
        self.dx = []
        self.dy = []
        self.yaw = []
        self.uv = []
        self.uw = []

        self.tState = []
        self._trstate = torch.zeros((self.K1 * self.nAgent, 8)).to(self.device).double()
        self._tlidarpt = torch.zeros((self.K1 * self.nAgent, 1, 360)).to(self.device).double()

        if self.writeTMode:
            self.writeTrainInfo()

    def clear(self):
        for i in self.replayMemory:
            i.clear()

    @preprocessState
    def ppState(self, obs):
        """
        wrapper를 통해 전처리 된 observation을 tuple형태로 반환한다.
        """
        return tuple([obs])

    def genOptim(self):
        """
        optimizer를 configuration에 맞춰 반환한다.
        """
        optimKeyList = list(self.optimData.keys())
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.agent.actor.buildOptim())
            
            if optimKey == 'critic':
                self.cOptim = getOptim(self.optimData[optimKey], self.agent.critic.buildOptim())
                
    def zeroGrad(self):
        """
        gradient를 zero로 반환
        """
        self.aOptim.zero_grad()
        self.cOptim.zero_grad()
    
    def np2tensor(self, listable):

        listable = list(listable)

        for i in range(len(listable)):
            listable[i] = torch.from_numpy(listable[i]).to(self.device).double()
        
        listable = tuple(listable)
        return listable

    def getAction(self, state, dMode=False):
        """
        action을 구한다. 이때 action을 생성하는 것은, oldAgent이다.
        """
        with torch.no_grad():
            if dMode:
                pass
            else:
                action = \
                    self.oldAgent.actorForward(self.np2tensor(state))
            action = action.cpu().numpy()
        return action
    
    def getActionHybridPolicy(self, state):

        with torch.no_grad():
            rstate = state[0]
            obs = state[1].cpu().numpy()
            actions = self.getAction(state)
            i = 0
            for rs, ob in zip(rstate, obs):
                dx = rs[0].item()
                dy = rs[1].item()
                yaw = -rs[4].item()
                distToGoal = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
                obs_dist = ob * self.data['lidar_roi_dist']
                if np.min(obs_dist) > self.data['r_safe'] or np.min(obs_dist) > distToGoal:
                    # print("--- pid mode ---")
                    uv_pid, uw_pid = self.pid.pid_policy(dx, dy, yaw)
                    action = np.array([uv_pid, -uw_pid])
                    actions[i] = action
                i += 1

        return actions

    def safety_policy(self, state):
        state[0, 6:] = state[0, 6:] / self.parm['p_scale']
        return state

    def step(self, step, epoch):
        """
        gradient를 바탕으로 weight를 update.
        """
        self.agent.actor.clippingNorm(5000)
        self.aOptim.step()
        self.agent.critic.clippingNorm(5000)
        self.cOptim.step()

        normA = self.agent.actor.calculateNorm().cpu().detach().numpy()
        normC = self.agent.critic.calculateNorm().cpu().detach().numpy()

        if self.writeTMode:
            self.writer.add_scalar('Actor Gradient Mag', normA, step+epoch)
            self.writer.add_scalar('Critic Gradient Mag', normC, step+epoch)
        
    @preprocessBatch
    def train(
        self, 
        state,
        action,
        gT,
        gAE,
        critic,
        step,
        epoch
    ):
        """
        전처리 된 입력값들을 바탕으로 objective function을 구하고
        
        이 후, backpropagation이 이루어진다
        """

        lossC, minusObj, entropy = self.agent.calLoss(
            self.copyAgent,
            state,
            action.detach(),
            gT.detach(),
            critic.detach(),
            gAE.detach()
        )

        objectFunction = minusObj + lossC
        objectFunction.backward()
        obj = minusObj.cpu().sum().detach().numpy()
        lossC = lossC.cpu().sum().detach().numpy()
        loss = lossC - obj

        if self.writeTMode:
            self.writer.add_scalar('Loss', loss, step+epoch)
            self.writer.add_scalar('Obj', -obj, step+epoch)
            self.writer.add_scalar('Critic Loss', lossC, step+epoch)
            entropy = entropy.detach().cpu().numpy()
            self.writer.add_scalar("Entropy", entropy, step+epoch)
            self.writer.add_scalar("gT", gT.mean(), step+epoch)
            self.writer.add_scalar("gAE", gAE.mean(), step+epoch)
        del loss
        del minusObj

    def getReturn(self, reward, critic, nCritic, done, Step_Agent=False):
        """
        GAE, rewards-to-go를 구하기 위한 method이다.
        input:
            reward:[np.array]  
                shape:[step, nAgent]
            done:[np.array]
                shape:[step, nAgent]
            critic:[tensor]
                shape[step*nAgent, 1]
        """
        nAgent = int(self.nAgent/self.div)
        gT, gAE = [], []
        step = len(reward)
        critic = critic.view((step, -1))
        nCritic = nCritic.view((step, -1))
        for i in range(nAgent):
            rA = reward[:, i]  # 160
            dA = done[:, i]  # 160
            cA = critic[:, i] 
            ncA = nCritic[:, i] 
            GT = []
            GTDE = []
            discounted_Td = 0
            discounted_r = ncA[-1]

            for r, is_terminal, c, nc in zip(
                    reversed(rA), 
                    reversed(dA), 
                    reversed(cA),
                    reversed(ncA)):
                td_error = r + self.gamma * nc - c
                discounted_r = r + self.gamma * discounted_r
                discounted_Td = td_error + self.gamma * self.labmda * discounted_Td
                GT.append(discounted_r)
                GTDE.append(discounted_Td)
            GT = torch.tensor(GT[::-1]).view((-1, 1)).to(self.device)
            GTDE = torch.tensor(GTDE[::-1]).view((-1, 1)).to(self.device)
            gT.append(GT)
            gAE.append(GTDE)

        gT = torch.cat(gT, dim=0)
        gAE = torch.cat(gAE, dim=0)

        if Step_Agent:
            gT.view(-1, 1)
            gAE.view(-1, 1)
        else:
            gT = gT.view(nAgent, -1)
            gT = gT.permute(1, 0).contiguous()
            gT = gT.view((-1, 1))

            gAE = gAE.view(nAgent, -1)
            gAE = gAE.permute(1, 0).contiguous()
            gAE = gAE.view((-1, 1))

        return gT, gAE
    
    def _step(env, behaviorNames, action):
        env.set_actions(behaviorNames, action)
        env.step()

    def getObs(self, init=False):
        """
        this method is for receiving messages from unity environment.
        args:
            init:[bool]
                :if init is true, the output only includes the observation.
        output:
            obsState:[np.array]
                shape:[nAgnet, 1447]
            reward:[np.array]
                shape:[nAgent, 1]
            done:[np.array]
                shape[:nAgent, 1]
        """
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        # image = decisionStep.obs[0]
        obs = decisionStep.obs[0]
        rewards = decisionStep.reward
        obs = np.array(obs)

        done = []
        
        for done_idx in obs[:, -1]:
            done.append(done_idx == 1)
        reward = rewards
        
        obsState = (obs, 0)

        if init:
            return obsState
        else:
            return(obsState, reward, done)
    
    def checkStep(self, action):
        """
        this method is for sending messages for unity environment.
        input:
            action:
                dtype:np.array
                shape:[nAgent, 2]
        """
        self.env.set_actions(self.behaviorNames, action)
        self.env.step()

    def evaluate(self):
        """
        evaluate를 통해 해당 알고리즘의 성능을 구한다.
        """
        self.loadUnityEnv()
        episodeReward = []
        k = 0
        Rewards = np.zeros(self.nAgent)
        
        obs = self.getObs(init=True)
        stateT = self.ppState(obs)
        # action = self.getActionHybridPolicy(stateT)
        action = self.getAction(stateT)
        TotalTrial = np.zeros(self.nAgent)
        TotalSucess = np.zeros(self.nAgent)
        step = 0
        while 1:
            self.checkStep(action)
            obs, reward, done = self.getObs()
            for k, r in enumerate(reward):
                if r > 3:
                    TotalSucess[k] += 1
                    TotalTrial[k] += 1

            Rewards += reward
            nStateT = self.ppState(obs)
            # nAction = self.getActionHybridPolicy(nStateT)
            nAction = self.getAction(nStateT)
            for i, d in enumerate(done):
                if d:
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0
                    TotalTrial[i] += 1
                    self.oldAgent.actor.zeroCellStateAgent(i)

            action = nAction
            stateT = nStateT
            step += 1
            
            if step % 3000 == 0:
                episodeReward = np.array(Rewards)
                reward = episodeReward.mean()
                SuccessRate = TotalSucess.sum()/TotalTrial.sum()
                SuccessRate = SuccessRate.mean()
                print("""
                Step : {:5d} // Reward : {:.3f}  // SuccessRate: {:.3f}
                """.format(step, reward, SuccessRate))
                print(TotalTrial.sum())
                print(TotalSucess.sum())
                episodeReward = []

    def run(self):
        self.loadUnityEnv()
        self.env.reset()
        episodeReward = []
        k = 0
        Rewards = np.zeros(self.nAgent)
        obs = self.getObs(init=True)
        stateT = self.ppState(obs)
        action = self.getAction(stateT)
        
        step = 0
        SamplingTime = 0
        TrainingTime = 0
        pid = os.getpid()
        current_process = psutil.Process(pid)
        while 1:
            # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
            # action을 환경으로 보내준다.
            self.checkStep(action)
            t = time.time()
            # 이를 바탕으로 환경으로부터 observation을 구한다.
            obs, reward, done = self.getObs()
            
            # reward logging
            Rewards += reward

            # observation을 전처리한후, 다음 행동을 구한다.
            nStateT = self.ppState(obs)
            nAction = self.getAction(nStateT)
            u = 0
            # inferencemode가 아니라면, replaymemory에  s, a, r ,s_, d를 추가한다.
            if self.inferMode is False:
                for z in range(self.div):
                    uu = u + int(self.nAgent/self.div)
                    self.replayMemory[z].append(
                            (stateT, action.copy(),
                                reward.copy()*self.rScaling, nStateT,
                                done.copy()))
                    self.ReplayMemory_Trajectory.append(
                            copy.deepcopy(stateT))
                    u = uu

            SamplingTime += (time.time() - t)

            # 초기화를 통해 sampling을 계속 진행시킨다.
            action = nAction
            stateT = nStateT
            step += 1

            # decayiong Log STd
            self.agent.decayingLogStd(step)
            self.oldAgent.decayingLogStd(step)
            self.copyAgent.decayingLogStd(step)

            # training
            if (step) % (self.updateStep) == 0 and self.inferMode is False:
                k += 1
                t = time.time()
                self.train(step, self.epoch)
                TrainingTime += (time.time() - t)
                # self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    self.copyAgent.update(self.agent)
                    k = 0

            # episode가 끝나면 lstm의 cell state를 초기화 한다.
            if True in done:
                self.agent.actor.zeroCellState()
                self.oldAgent.actor.zeroCellState()
                self.copyAgent.actor.zeroCellState()
                self.ReplayMemory_Trajectory.clear()
                self.env.step()

                obs = self.getObs(init=True)
                stateT = self.ppState(obs)
                action = self.getAction(stateT)

                # 환경 역시 초기화를 위해 한 스텝 이동한다.

            # 10000 step마다 결과를 print, save한다.
            if step % 1280 == 0:
                episodeReward = np.array(Rewards)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:.3f}
                SamplingTime : {:.3f} // TrainingTime : {:.3f}
                """.format(step, reward, SamplingTime, TrainingTime))
                Rewards = np.zeros(self.nAgent)
                SamplingTime = 0
                TrainingTime = 0
                if (reward > self.RecordScore):
                    self.RecordScore = reward
                    sPath = './save/PPO/'+self.data['envName']+str(self.RecordScore)+'.pth'
                    torch.save(self.agent.state_dict(), sPath)
                torch.save(self.agent.state_dict(), self.sPath)
