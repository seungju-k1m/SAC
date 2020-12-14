import torch
import datetime
import numpy as np
from baseline.baseTrainer import ONPolicy
from PPO.Agent import ppoAgent
from baseline.utils import getOptim
from collections import deque


def preprocessBatch(f):
    def wrapper(self, step, epoch):
        k1 = 160
        k2 = 10
        div = int(k1/k2)
        rstate, action, reward, done = \
            [], [], [], []
        tState = []
        for data in self.replayMemory[0]:
            s, a, r, ns, d = data
            rstate.append(s)
            action.append(a)
            reward.append(r)
            done.append(d)
        for data in self.ReplayMemory_Trajectory:
            ts = data
            tState.append(ts)
        if len(tState) == k1:
            zeroMode = True
        else:
            tState = torch.cat(tState[:-k1], dim=0)
            zeroMode = False
        state = torch.cat(rstate, dim=0)
        nstate = torch.cat((state, ns), dim=0)
        reward = np.array(reward)
        done = np.array(done)
        action = torch.tensor(action).to(self.device)

        self.agent.actor.zeroCellState()
        self.agent.critic.zeroCellState()
        self.copyAgent.actor.zeroCellState()
        self.copyAgent.critic.zeroCellState()

        if zeroMode == False:
            self.agent.critic.forward(tuple([tState]))
            self.copyAgent.critic.forward(tuple([tState]))
            self.agent.actor.forward(tuple([tState]))
            self.copyAgent.actor.forward(tuple([tState]))

        # 1. calculate the target value for actor and critic
        self.agent.actor.detachCellState()
        InitActorCellState = self.agent.actor.getCellState()
        InitCopyActorCellState = self.copyAgent.actor.getCellState()

        self.agent.critic.detachCellState()
        InitCriticCellState = self.agent.critic.getCellState()
        InitCopyCriticCellState = self.copyAgent.critic.getCellState()

        self.agent.critic.setCellState(InitCriticCellState)
        
        # 2. implemented the training using the truncated BPTT
        for _ in range(epoch):
            self.agent.actor.setCellState(InitActorCellState)
            value = self.agent.critic.forward(tuple([nstate]))[0]  # . step, nAgent, 1 -> -1, 1
            value = value.view(k1+1, self.nAgent, 1)
            nvalue = value[1:]
            value = value[:-1]
            gT, gAE = self.getReturn(reward, value, nvalue, done)
            gT = gT.view(k1, self.nAgent)
            gAE = gAE.view(k1, self.nAgent)

            self.agent.critic.setCellState(InitCriticCellState)
            self.copyAgent.actor.setCellState(InitCopyActorCellState)
            self.copyAgent.critic.setCellState(InitCopyCriticCellState)
            self.zeroGrad()
            for i in range(div):
                _state = tuple([state[i*k2:(i+1)*k2]])
                _action = action[i*k2:(i+1)*k2].view((-1, 2))
                _gT = gT[i*k2:(i+1)*k2].view(-1, 1)
                _gAE = gAE[i*k2:(i+1)*k2].view(-1, 1)
                _value = value[i*k2:(i+1)*k2].view(-1, 1)
                f(self, _state, _action, _gT, _gAE, _value, step, epoch)
                self.agent.actor.detachCellState()
                self.agent.critic.detachCellState()
            self.step(step+i, epoch)
            self.agent.actor.zeroCellState()
            self.agent.critic.zeroCellState()
            if zeroMode == False:
                self.agent.critic.forward(tuple([tState]))
                self.agent.actor.forward(tuple([tState]))
            InitActorCellState = self.agent.actor.getCellState()
            InitCriticCellState = self.agent.critic.getCellState()

    return wrapper


def preprocessState(f):
    def wrapper(self, obs):
        rState = torch.tensor(obs[:, :6]).float().to(self.device)
        lidarPt = torch.tensor(obs[:, 8:8+self.sSize[-1]]).float().to(self.device)
        state = [torch.unsqueeze(torch.cat((rState, lidarPt), dim=1), dim=0)]
        return state
    return wrapper


class PPOOnPolicyTrainer(ONPolicy):

    def __init__(self, cfg):
        super(PPOOnPolicyTrainer, self).__init__(cfg)
        
        if 'fixedTemp' in self.keyList:
            self.fixedTemp = self.data['fixedTemp'] == "True"
            if self.fixedTemp:
                if 'tempValue' in self.keyList:
                    self.tempValue = self.data['tempValue']
            else:
                self.fixedTemp = False
                self.tempValue = self.agent.temperature

        if self.device != "cpu":
            self.device = torch.device(self.device)
            
        else:
            self.device = torch.device("cpu")
        self.entropyCoeff = self.data['entropyCoeff']
        self.epsilon = self.data['epsilon']
        self.labmda = self.data['lambda']
        initLogStd = torch.tensor(self.data['initLogStd']).to(self.device).float()
        finLogStd = torch.tensor(self.data['finLogStd']).to(self.device).float()
        annealingStep = self.data['annealingStep']
        self.LSTMNum = self.data['LSTMNum']
        self.agent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMNum=self.LSTMNum)
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
            LSTMNum=self.LSTMNum)
        self.oldAgent.to(self.device)
        self.oldAgent.update(self.agent)

        self.copyAgent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMNum=self.LSTMNum)
        self.copyAgent.to(self.device)
        self.copyAgent.update(self.agent)
        
        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.sPath += name + '_' + str(time)+'.pth'

        self.sSize = self.aData['sSize']
        self.updateStep = self.data['updateStep']
        self.genOptim()

        self.div = self.data['div']

        self.replayMemory = [deque(maxlen=self.updateStep) for i in range(self.div)]
        self.epoch = self.data['epoch']
        self.updateOldP = self.data['updateOldP']
        self.Number_Episode = 0
        self.Number_Sucess = 0

        self.ReplayMemory_Trajectory = deque(maxlen=1000000)

        if self.writeTMode:
            self.writeTrainInfo()
    
    def clear(self):
        for i in self.replayMemory:
            i.clear()

    @preprocessState
    def ppState(self, obs):
        return tuple([obs])

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.agent.actor.buildOptim())
            if optimKey == 'critic':
                self.cOptim = getOptim(self.optimData[optimKey], self.agent.critic.buildOptim())
                
    def zeroGrad(self):
        self.aOptim.zero_grad()
        self.cOptim.zero_grad() 

    def getAction(self, state, dMode=False):
        with torch.no_grad():
            if dMode:
                pass
            else:
                action = \
                    self.oldAgent.actorForward(state)
            action = action.cpu().numpy()
        return action

    def step(self, step, epoch):
        # self.agent.critic.clippingNorm(500)
        self.cOptim.step()
        self.agent.actor.clippingNorm(5)
        self.aOptim.step()

        normA = self.agent.actor.calculateNorm().cpu().detach().numpy()
        normC = self.agent.critic.calculateNorm().cpu().detach().numpy()

        if self.writeTMode:
            self.writer.add_scalar('Action Gradient Mag', normA, step+epoch)
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
        lossC = self.agent.calQLoss(
            state,
            gT.detach(),
        
        )
        lossC.backward()

        minusObj, entropy = self.agent.calAObj(
            self.copyAgent,
            state,
            action,
            gT.detach() - critic.detach()
        )
        minusObj.backward()

        obj = minusObj.cpu().sum().detach().numpy()
        lossC = lossC.cpu().sum().detach().numpy()
        loss = lossC - obj

        if self.writeTMode:
            self.writer.add_scalar('Loss', loss, step+epoch)
            self.writer.add_scalar('Obj', -obj, step+epoch)
            self.writer.add_scalar('Critic Loss', lossC, step+epoch)
            entropy = entropy.detach().cpu().numpy()
            self.writer.add_scalar("Entropy", entropy, step+epoch)

    def getReturn(self, reward, critic, nCritic, done, Step_Agent=False):
        """
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
            rA = reward[:, i]  # [step] , 100
            dA = done[:, i]  # [step] , 100
            cA = critic[:, i]
            ncA = nCritic[:, i] 
            GT = []
            GTDE = []
            discounted_Td = 0
            if dA[:-1][0]:
                discounted_r = 0
            else:
                discounted_r = cA[-1]
            for r, is_terminal, c, nc in zip(
                    reversed(rA), 
                    reversed(dA), 
                    reversed(cA),
                    reversed(ncA)):
                
                if is_terminal:
                    td_error = r - c
                else:
                    td_error = r + self.gamma * nc - c

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
        obsState = np.zeros((self.nAgent, 1447), dtype=np.float32)
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        obs, tobs = decisionStep.obs[0], terminalStep.obs[0]
        rewards, treward = decisionStep.reward, terminalStep.reward
        tAgentId = terminalStep.agent_id
        
        done = [False for i in range(self.nAgent)]
        reward = [0 for i in range(self.nAgent)]
        obsState = np.array(obs)
        reward = rewards
        
        # for i, state in zip(agentId, obs):
        #     state = np.array(state)
        #     obsState[i] = state
        #     done[i] = False
        #     reward[i] = rewards[k]
        #     k += 1
        k = 0
        for i, state in zip(tAgentId, tobs):
            state = np.array(state)
            obsState[i] = state
            done[i] = True
            self.Number_Episode += 1
            reward[i] = treward[k]
            if (reward[i]>1):
                self.Number_Sucess += 1
            k += 1
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
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        agentId = decisionStep.agent_id
        if len(agentId) != 0:
            self.env.set_actions(self.behaviorNames, action)
        self.env.step()
    
    def LogSucessRate(self, step):
        if self.writeTMode:
            self.writer.add_scalar("Sucess Rate", (self.Number_Sucess/self.Number_Episode), step)
            self.Number_Episode = 0
            self.Number_Sucess = 0

    def run(self):
        episodeReward = []
        k = 0
        Rewards = np.zeros(self.nAgent)
        
        obs = self.getObs(init=True)
        stateT = self.ppState(obs)
        action = self.getAction(stateT)
        step = 0
        while 1:
            self.checkStep(action)
            obs, reward, done = self.getObs()
            Rewards += reward
            nStateT = self.ppState(obs)
            nAction = self.getAction(nStateT)
            u = 0
            if self.inferMode == False:
                for z in range(self.div):
                    uu = u + int(self.nAgent/self.div)
                    self.replayMemory[z].append(
                            (stateT[0][u:uu], action[u:uu].copy(),
                             reward[u:uu]*self.rScaling, nStateT[0][u:uu],
                             done[u:uu].copy()))
                    self.ReplayMemory_Trajectory.append(
                            (stateT[0][u:uu]))
                    u = uu
            for i, d in enumerate(done):
                if d:
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0

            action = nAction
            stateT = nStateT
            step += 1
            self.agent.decayingLogStd(step)
            self.oldAgent.decayingLogStd(step)
            if (step) % (self.updateStep) == 0 and self.inferMode == False:
                k += 1
                self.train(step, self.epoch)
                self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    self.copyAgent.update(self.agent)
                    k = 0
            if True in done:
                self.agent.actor.zeroCellState()
                self.agent.critic.zeroCellState()
                self.oldAgent.actor.zeroCellState()
                self.oldAgent.critic.zeroCellState()
                self.copyAgent.actor.zeroCellState()
                self.copyAgent.critic.zeroCellState()
                self.ReplayMemory_Trajectory.clear()
                self.env.step()
            
            if step % 2000 == 0:
                self.LogSucessRate(step)
            
            if step % 2000 == 0:
                episodeReward = np.array(episodeReward)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:.3f}  
                """.format(step, reward))
                episodeReward = []
                torch.save(self.agent.state_dict(), self.sPath)