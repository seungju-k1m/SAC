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
        self.updateOldP = self.data['updateOldP']

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
            rState, lidarPt = obs[:6], obs[7:127]
            rState = torch.tensor(rState).view((1, -1))
            lidarPt = torch.tensor(lidarPt).view((1, -1))

            return (rState, lidarPt)

    def annealingLogStd(self, step):
        alpha = step / self.annealingStep
        temp = (1 - alpha) * self.initLogStd + alpha * self.finLogStd
        self.agent.logStd = temp
        self.oldAgent.logStd = temp

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.CNN = self.agent.CNN.to(self.device)
        self.CNNF = self.agent.CNNF.to(self.device)
        self.LSTM = self.agent.LSTM.to(self.device)
        self.actor = self.agent.actor.to(self.device)
        self.critic = self.agent.critic.to(self.device)
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                # self.aOptim = getOptim(self.optimData[optimKey], self.actor)ß
                # self.cnnOptim = getOptim(self.optimData[optimKey], self.CNN)
                # self.lOptim = getOptim(self.optimData[optimKey], self.LSTM)
                # self.aOptim = getOptim(self.optimData[optimKey], (self.actor))
                self.aOptim = getOptim(self.optimData[optimKey], (self.actor, self.LSTM, self.CNN))

            if optimKey == 'critic':
                # self.cOptim = getOptim(self.optimData[optimKey], self.critic)
                # self.cOptim = getOptim(self.optimData[optimKey], (self.critic, self.LSTM, self.CNN))
                self.cOptim = getOptim(self.optimData[optimKey], (self.critic, self.CNNF))

    def zeroGrad(self):
        self.aOptim.zero_grad()
        # self.cnnOptim.zero_grad()
        self.cOptim.zero_grad()
        # self.lOptim.zero_grad()

    def getAction(self, state, oldLstmState=None, lstmState=None, dMode=False):
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
        rState, lidarPt = [], []
        for r, l in state:
            rState.append(r)
            lidarPt.append(l)
        rState = torch.cat(rState, dim=0).to(self.device)
        lidarPt = torch.cat(lidarPt, dim=0).to(self.device)
        bSize = rState.shape[0]
        state = (rState, lidarPt)

        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            lstmState = (hAState, cAState)

        if oldLstmState is None:
            ohAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            ocAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            oldLstmState = (ohAState, ocAState)

        with torch.no_grad():
            if dMode:
                pass
            else:
                action, oldLstmState = \
                    self.oldAgent.actorForward(state, lstmState=oldLstmState)
                
                _, lstmState = \
                    self.agent.actorForward(state, lstmState=lstmState)
            action = action.cpu().numpy()
        return action, oldLstmState, lstmState

    def train(self, step, k, epoch):
        """
        this method is for training!!

        before training, shift the device of idx network and data

        """
        rState, lidarPt, hA, cA,  actions, rewards, dones = \
            [], [], [], [], [], [], []
        nrState, nlidarPt, nhA, ncA = \
            [], [], [], []
        for data in self.replayMemory[k]:
            state = data[0]
            state, (h, c) = state
            for r, l in state:
                rState.append(r)
                lidarPt.append(l)
            hA.append(h)
            cA.append(c)

            actions.append(data[1])
            rewards.append(data[2])

            dones.append(data[4])

            nstate = data[3]
            nstate, (nh, nc) = nstate
            for r, l in nstate:
                nrState.append(r)
                nlidarPt.append(l)
            nhA.append(nh)
            ncA.append(nc)

        rState = torch.cat(rState, dim=0).to(self.device).detach()  # states step, nAgent
        lidarPt = torch.cat(lidarPt, dim=0).to(self.device).detach()
        lidarPt = torch.unsqueeze(lidarPt, dim=1)
        states = (rState, lidarPt)
        hA = torch.cat(hA, dim=1).to(self.device).detach()
        cA = torch.cat(cA, dim=1).to(self.device).detach()

        nrState = torch.cat(nrState, dim=0).to(self.device).detach()
        nlidarPt = torch.cat(nlidarPt, dim=0).to(self.device).detach()
        nlidarPt = torch.unsqueeze(nlidarPt, dim=1)
        nstates = (nrState, nlidarPt)
        nhA = torch.cat(nhA, dim=1).to(self.device).detach()
        ncA = torch.cat(ncA, dim=1).to(self.device).detach()
        
        lstmState = (hA, cA)
        nlstmState = (nhA, ncA)

        actions = torch.tensor(actions).to(self.device).view((-1, 2))
        rewards = np.array(rewards)
        dones = np.array(dones)

        with torch.no_grad():
            critic = self.oldAgent.criticForward(states)
            nCritic = self.oldAgent.criticForward(nstates)

        gT, gAE = self.getReturn(rewards, critic, nCritic, dones)  # step, nAgent
        
        self.zeroGrad()
        lossC = self.agent.calQLoss(
            states,
            gT.detach(),
            
        )
        lossC.backward()
        
        self.cOptim.step()
        normC = calGlobalNorm(self.critic) + calGlobalNorm(self.CNNF)
        self.zeroGrad()

        minusObj, entropy = self.agent.calAObj(
            self.oldAgent,
            states,
            lstmState,
            actions,
            gAE.detach()
        )
        minusObj.backward()

        # self.cnnOptim.step()
        self.aOptim.step()
        # self.lOptim.step()
        
        aN = calGlobalNorm(self.actor)
        aF1N = calGlobalNorm(self.LSTM) + calGlobalNorm(self.CNN)
        normA = aN + aF1N 

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
            entropy = entropy.detach().cpu().numpy()
            self.writer.add_scalar("Entropy", entropy, step+epoch)

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
            aV = state[4]
            # print(aV)
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
        k = 0
        Rewards = np.zeros(self.nAgent)
        
        obs = self.getObs(init=True)
        stateT = []
        for b in range(self.nAgent):
            ob = obs[b]
            state = self.ppState(ob)
            stateT.append(state)
        # stateT = torch.stack(stateT, dim=0)
        lstmState = self.zeroLSTMState()
        action, onlstmState, nlstmState = self.getAction(
            stateT, lstmState=lstmState, oldLstmState=lstmState)
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
            # nStateT = torch.stack(nStateT, dim=0).to(self.device)
            nAction, onnlstmState, nnlstmState = self.getAction(
                nStateT, lstmState=nlstmState, oldLstmState=onlstmState)
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
                    nnlstmState = self.setZeroLSTMState(nnlstmState, i)
                    onnlstmState = self.setZeroLSTMState(onnlstmState, i)
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0

            action = nAction
            stateT = nStateT
            lstmState = nlstmState

            nlstmState = nnlstmState
            onlstmState = onnlstmState

            step += 1
            self.annealingLogStd(step)
            if step % self.updateStep == 0:
                k += 1
                for epoch in range(self.epoch):
                    for j in range(self.div):
                        self.train(step, j, epoch)
                self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    k = 0
            
            if step % 3000 == 0:
                episodeReward = np.array(episodeReward)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:3f}  
                """.format(step, reward))
                episodeReward = []
                torch.save(self.agent.state_dict(), self.sPath)

