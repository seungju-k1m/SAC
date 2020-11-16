import math
import torch
import numpy as np
from baseline.baseTrainer import ONPolicy
from PPO.Agent import ppoAgent
from baseline.utils import getOptim, calGlobalNorm
from collections import deque


class PPOOnPolicyTrainer(ONPolicy):

    def __init__(self, cfg):
        super(PPOOnPolicyTrainer, self).__init__(cfg)
        if self.lPath != "None":
            self.agent.load_state_dict(
                torch.load(self.lPath, map_location=self.device)
            )
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

        self.initLogStd = self.data['initLogStd']
        self.LogStd = self.initLogStd
        self.finLogStd = self.data['finLogStd']
        self.annealingStep = self.data['annealingStep']

        self.fixedSigma = self.data['fixedSigma'] == 'True'
        self.agent = ppoAgent(
            self.aData, 
            self.optimData, 
            coeff=self.entropyCoeff,
            logStd=self.LogStd,
            fixedSigma=self.fixedSigma,
            epsilon=self.epsilon)
        self.agent.to(self.device)
        self.oldAgent = ppoAgent(
            self.aData,
            self.optimData,
            coeff=self.entropyCoeff,
            logStd=self.LogStd,
            fixedSigma=self.fixedSigma,
            epsilon=self.epsilon)
        self.oldAgent.to(self.device)
        self.oldAgent.update(self.agent)
 
        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        self.labmda = self.data['Lambda']
        self.sPath += name + '_' + str(self.nAgent) + '_LSTMV1_.pth'

        self.sSize = self.aData['sSize']
        self.hiddenSize = self.aData['LSTM']['hiddenSize']
        self.updateStep = self.data['updateStep']
        self.genOptim()

        self.div = self.data['div']

        self.replayMemory = [deque(maxlen=self.updateStep) for i in range(self.div)]
        self.epoch = self.data['epoch']

        if self.writeTMode:
            self.writeTrainInfo()

    def writeTrainInfo(self):
        super(PPOOnPolicyTrainer, self).writeTrainInfo()
        print(self.info)

        self.writer.add_text('info', self.info, 0)    
    
    def clear(self):
        for i in self.replayMemory:
            i.clear()

    def ppState(self, obs):
        """
        input:
            obs:[np.array]
                shpae:[1447,]
        """
        if self.imgMode:
            rState, targetOn, lidarPt = obs[:6], obs[6], obs[7:]
            targetPos = np.reshape(rState[:2], (1, 2))
            rState = torch.tensor(rState).to(self.device)
            lidarImg = torch.zeros(self.sSize[1]**2).to(self.device)
            lidarPt = lidarPt[lidarPt != 0]
            lidarPt -= 1000
            lidarPt = np.reshape(lidarPt, (-1, 2))
            R = [[math.cos(rState[-1]), -math.sin(rState[-1])], [math.sin(rState[-1]), math.cos(rState[-1])]]
            R = np.array(R)
            lidarPt = np.dot(lidarPt, R)
            locXY = (((lidarPt + 7) / 14) * (self.sSize[-1]+1)).astype(np.int16)
            locYX = locXY[:, ::-1]
            locYX = np.unique(locYX, axis=0)
            locYX = locYX[:, 0] * self.sSize[1] + locYX[:, 1]
            locYX = np.clip(locYX, self.sSize[1], self.sSize[1]**2 - 2)
            lidarImg[locYX] = 1
            lidarImg = lidarImg.view(self.sSize[1:])
            if targetOn == 1:
                pt = np.dot(targetPos, R)[0]
                locX = int(((pt[0]+7) / 14)*self.sSize[-1])
                locY = int(((pt[1]+7) / 14)*self.sSize[-1])
                if locX == (self.sSize[-1]-1):
                    locX -= 1
                if locY == (self.sSize[-1]-1):
                    locY -= 1
                lidarImg[locY+1, locX+1] = 10
            lidarImg = torch.unsqueeze(lidarImg, dim=0)
            lidarImg[0, 0, :6] = rState
            return lidarImg
        else:
            rState, lidarPt = obs[:6], obs[7:727]
            rState = torch.tensor(rState).view((1, -1))
            lidarPt = torch.tensor(lidarPt).view((1, -1))

            state = torch.cat((rState, lidarPt), dim=1).float().to(self.device)

            return state

    def annealingLogStd(self, step):
        alpha = step / self.annealingStep
        temp = (1 - alpha) * self.initLogStd + alpha * self.finLogStd
        self.agent.logStd = temp
        self.oldAgent.logStd = temp

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.CNN = self.agent.CNN.to(self.device)
        self.LSTM = self.agent.LSTM.to(self.device)
        self.actor = self.agent.actor.to(self.device)
        self.critic = self.agent.critic.to(self.device)
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.cnnOptim = getOptim(self.optimData[optimKey], self.CNN)
                self.lOptim = getOptim(self.optimData[optimKey], self.LSTM)

            if optimKey == 'critic':
                self.cOptim = getOptim(self.optimData[optimKey], self.critic)

    def zeroGrad(self):
        self.aOptim.zero_grad()
        self.cnnOptim.zero_grad()
        self.cOptim.zero_grad()
        self.lOptim.zero_grad()

    def getAction(self, state, lstmState=None, dMode=False):
        """
        input:
            state:
                dtype:tensor
                shape:[nAgent, 726]
            lstmState:
                dtype:tuple, (hA, cA)
                shape:[1, nAgent, hiddenSize] for each state
        output:
            action:
                dtype:np.array
                shape[nAgnet, 2]
            lstmState:
                dtype:tuple ((tensor, tensor), (tensor, tensor), (tensor, tensor))
                shape:[1, nAgent, 512] for each state
        """
        bSize = state.shape[0]

        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            lstmState = (hAState, cAState)

        with torch.no_grad():
            if dMode:
                pass
            else:
                action, lstmState = \
                    self.agent.actorForward(state, lstmState=lstmState)
            action = action.cpu().numpy()
        return action, lstmState

    def train(self, step, k, epoch):
        """
        this method is for training!!

        before training, shift the device of idx network and data

        """
        states, hA, cA,  actions, rewards, dones = \
            [], [], [], [], [], []
        nstates, nhA, ncA = \
            [], [], []
        for data in self.replayMemory[k]:
            states.append(data[0][0])
            hA.append(data[0][1][0])
            cA.append(data[0][1][1])

            actions.append(data[1])
            rewards.append(data[2])

            dones.append(data[4])

            nstates.append(data[3][0])
            nhA.append(data[3][1][0])
            ncA.append(data[3][1][1])

        states = torch.cat(states, dim=0).to(self.device)  # states step, nAgent
        hA = torch.cat(hA, dim=1).to(self.device).detach()
        cA = torch.cat(cA, dim=1).to(self.device).detach()

        nstates = torch.cat(nstates, dim=0).to(self.device)
        nhA = torch.cat(nhA, dim=1).to(self.device).detach()
        ncA = torch.cat(ncA, dim=1).to(self.device).detach()
        
        lstmState = (hA, cA)
        nlstmState = (nhA, ncA)

        actions = torch.tensor(actions).to(self.device).view((-1, 2))
        rewards = np.array(rewards)
        dones = np.array(dones)

        with torch.no_grad():
            critic = self.oldAgent.criticForward(states, lstmState)
            nCritic = self.oldAgent.criticForward(nstates, nlstmState)

        gT, gAE = self.getReturn(rewards, critic, nCritic, dones)  # step, nAgent
        
        self.zeroGrad()
        lossC = self.agent.calQLoss(
            states.detach(),
            lstmState,
            gT.detach(),
            
        )

        minusObj= self.agent.calAObj(
            self.oldAgent,
            states.detach(),
            lstmState,
            actions,
            gAE.detach()-critic
        )
        minusObj.backward()
        lossC.backward()
        self.cOptim.step()
        self.cnnOptim.step()
        self.aOptim.step()
        self.lOptim.step()
        
        aN = calGlobalNorm(self.actor)
        aF1N = calGlobalNorm(self.LSTM) + calGlobalNorm(self.CNN)
        normA = aN + aF1N 

        cN = calGlobalNorm(self.critic)
        normC = cN

        norm = normA + normC
        obj = minusObj.cpu().sum().detach().numpy()
        lossC = lossC.cpu().sum().detach().numpy()
        loss = lossC - obj

        if self.writeTMode:
            self.writer.add_scalar('Action Gradient Mag', normA, step+epoch)
            self.writer.add_scalar('Critic Gradient Mag', normC, step+epoch)
            self.writer.add_scalar('Gradient Mag', norm, step+epoch)
            self.writer.add_scalar('Loss', loss, step+epoch)
            self.writer.add_scalar('Obj', -obj, step+epoch)
            self.writer.add_scalar('Critic Loss', lossC, step+epoch)

    def getReturn(self, reward, critic, nCritic, done):
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
            if dA[-1]:
                discounted_r = 0
            else:
                discounted_r = cA[-1]
            for r, is_terminal, c, nc in zip(
                    reversed(rA), 
                    reversed(dA), 
                    reversed(cA),
                    reversed(ncA)):

                if is_terminal:
                    discounted_r = 0
                    discounted_Td = 0
                    td_error = r - c
                else:
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
        gT = gT.view(nAgent, -1)
        gT = gT.permute(1, 0).contiguous()
        gT = gT.view((-1, 1))

        gAE = torch.cat(gAE, dim=0)
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
        obsState = np.zeros((self.nAgent, self.obsShape), dtype=np.float32)
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        obs, tobs = decisionStep.obs[0], terminalStep.obs[0]
        rewards, treward = decisionStep.reward, terminalStep.reward
        tAgentId = terminalStep.agent_id
        agentId = decisionStep.agent_id
        
        done = [False for i in range(self.nAgent)]
        reward = [0 for i in range(self.nAgent)]
        k = 0
        
        for i, state in zip(agentId, obs):
            state = np.array(state)
            obsState[i] = state
            done[i] = False
            reward[i] = rewards[k]
            k += 1
        k = 0
        for i, state in zip(tAgentId, tobs):
            state = np.array(state)
            obsState[i] = state
            done[i] = True
            reward[i] = treward[k]
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

    def zeroLSTMState(self):
        hAState = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)
        cAState = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)

        lstmState = (hAState, cAState)
        return lstmState
    
    def setZeroLSTMState(self, lstmState, idx):
        ha, ca = lstmState

        hAState = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        cAState = torch.zeros(1, 1, self.hiddenSize).to(self.device)

        ha[:, idx:idx+1, :] = hAState
        ca[:, idx:idx+1, :] = cAState

        lstmState = (ha, ca)

        return lstmState

    def run(self):
        episodeReward = []
        Rewards = np.zeros(self.nAgent)
        
        obs = self.getObs(init=True)
        stateT = []
        for b in range(self.nAgent):
            ob = obs[b]
            state = self.ppState(ob)
            stateT.append(state)
        stateT = torch.stack(stateT, dim=0)
        lstmState = self.zeroLSTMState()
        action, nlstmState = self.getAction(stateT, lstmState=lstmState)
        step = 1
        while 1:
            self.checkStep(action)
            obs, reward, done = self.getObs()
            Rewards += reward
            nStateT = []

            for b in range(self.nAgent):
                ob = obs[b]
                state = self.ppState(ob)
                nStateT.append(state)
            nStateT = torch.stack(nStateT, dim=0).to(self.device)
            nAction, nnlstmState = self.getAction(nStateT, lstmState=nlstmState)
            u = 0
            for z in range(self.div):
                uu = u + int(self.nAgent/self.div)
                temp = (lstmState[0][:, u:uu], lstmState[1][:, u:uu])
                ntemp = (nlstmState[0][:, u:uu], nlstmState[1][:, u:uu])
                self.replayMemory[z].append(
                    ((stateT[u:uu], temp), action[u:uu].copy(),
                        reward[u:uu]*self.rScaling, (nStateT[u:uu], ntemp),
                        done[u:uu].copy())
                )
                u = uu
            for i, d in enumerate(done):
                if d:
                    nlstmState = self.setZeroLSTMState(nlstmState, i)
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0

            action = nAction
            stateT = nStateT
            lstmState = nlstmState
            nlstmState = nnlstmState

            step += 1
            self.annealingLogStd(step)
            if step % self.updateStep == 0:
                for epoch in range(self.epoch):
                    for j in range(self.div):
                        self.train(step, j, epoch)
                self.oldAgent.update(self.agent)
                self.clear()
            
            if step % 400 == 0:
                episodeReward = np.array(episodeReward)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:3f}  
                """.format(step, reward))
                episodeReward = []
                torch.save(self.agent.state_dict(), self.sPath)

