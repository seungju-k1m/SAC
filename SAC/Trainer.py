import torch
import numpy as np
from baseline.baseTrainer import ONPolicy
from SAC.Agent import sacAgent
from baseline.utils import getOptim, calGlobalNorm


class sacOnPolicyTrainer(ONPolicy):
    def __init__(self, cfg):
        super(sacOnPolicyTrainer, self).__init__(cfg)
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
        self.agent = sacAgent(self.aData, self.optimData)
        self.agent.to(self.device)
        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        self.sPath += name + '_' + str(self.nAgent)
        if self.fixedTemp:
            self.sPath += str(int(self.tempValue * 100)) + '.pth'
        else:
            self.sPath += '.pth'
        self.sSize = self.aData['sSize']
        self.hiddenSize = self.aData['actorFeature01']['hiddenSize']
        self.updateStep = self.data['updateStep']
        self.genOptim()

    def ppState(self, obs):
        """
        input:
            obs:[np.array]
                shpae:[1447,]
        output:
            lidarImg:[tensor. device]
                shape:[1, 96, 96]
        """
        rState, lidarPt = obs[:6], obs[7:720+7]
        state = np.concatenate((rState, lidarPt), axis=0).reshape((1, -1))
        state = torch.tensor(state).float().to(self.device)
        return state

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.actor, self.actorFeature01 = self.agent.actor, self.agent.actorFeature01
        self.criticFeature01, self.critic01 = self.agent.criticFeature01, self.agent.critic01
        self.criticFeature02,  self.critic02 = self.agent.criticFeature02, self.agent.critic02

        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.aFOptim01 = getOptim(self.optimData[optimKey], self.actorFeature01)
            if optimKey == 'critic':
                self.cOptim01 = getOptim(self.optimData[optimKey], self.critic01)
                self.cFOptim01 = getOptim(self.optimData[optimKey], self.criticFeature01)
                self.cOptim02 = getOptim(self.optimData[optimKey], self.critic02)
                self.cFOptim02 = getOptim(self.optimData[optimKey], self.criticFeature02)
            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(self.optimData[optimKey], [self.tempValue], floatV=True)
                 
    def zeroGrad(self):
        self.aOptim.zero_grad()
        self.aFOptim01.zero_grad()

        self.cOptim01.zero_grad()
        self.cFOptim01.zero_grad()

        self.cOptim02.zero_grad()
        self.cFOptim02.zero_grad()

    def getAction(self, state, lstmState=None, dMode=False):
        """
        input:
            state:
                dtype:tensor
                shape:[nAgent, 726]
            lstmState:
                dtype:tuple, ((hA, cA), (hC1, cC1), (hC2, cC2))
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
            hCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)

            lstmState = ((hAState, cAState), (hCState01, cCState01), (hCState02, cCState02))

        with torch.no_grad():
            if dMode:
                pass
            else:
                action, _, __, ___, lstmState = \
                    self.agent.forward(state, lstmState=lstmState)
            action = action.cpu().numpy()
        return action, lstmState

    def train(self, step):
        """
        this method is for training!!

        before training, shift the device of idx network and data

        """
        states, hA, cA, hC1, cC1, hC2, cC2, actions, rewards, dones = \
            [], [], [], [], [], [], [], [], [], []
        nstates, nhA, ncA, nhC1, ncC1, nhC2, ncC2 = \
            [], [], [], [], [], [], []
        for data in self.replayMemory:
            states.append(data[0][0])
            hA.append(data[0][1][0][0])
            cA.append(data[0][1][0][1])
            hC1.append(data[0][1][1][0])
            cC1.append(data[0][1][1][1])
            hC2.append(data[0][1][2][0])
            cC2.append(data[0][1][2][1])

            actions.append(data[1])
            rewards.append(data[2])

            nstates.append(data[3][0])
            nhA.append(data[3][1][0][0])
            ncA.append(data[3][1][0][1])
            nhC1.append(data[3][1][1][0])
            ncC1.append(data[3][1][1][1])
            nhC2.append(data[3][1][2][0])
            ncC2.append(data[3][1][2][1])

            dones.append(data[4])
        
        states = torch.cat(states, dim=0).to(self.device)
        hA = torch.cat(hA, dim=1).to(self.device).detach()
        cA = torch.cat(cA, dim=1).to(self.device).detach()
        hC1 = torch.cat(hC1, dim=1).to(self.device).detach()
        cC1 = torch.cat(cC1, dim=1).to(self.device).detach()
        hC2 = torch.cat(hC2, dim=1).to(self.device).detach()
        cC2 = torch.cat(cC2, dim=1).to(self.device).detach()

        nstates = torch.cat(nstates, dim=0).to(self.device)
        nhA = torch.cat(nhA, dim=1).to(self.device).detach()
        ncA = torch.cat(ncA, dim=1).to(self.device).detach()
        nhC1 = torch.cat(nhC1, dim=1).to(self.device).detach()
        ncC1 = torch.cat(ncC1, dim=1).to(self.device).detach()
        nhC2 = torch.cat(nhC2, dim=1).to(self.device).detach()
        ncC2 = torch.cat(ncC2, dim=1).to(self.device).detach()
        
        lstmState = ((hA, cA), (hC1, cC1), (hC2, cC2))
        nlstmState = ((nhA, ncA), (nhC1, ncC1), (nhC2, ncC2))

        actions = torch.tensor(actions).to(self.device).view((-1, 2))
        rewards = np.array(rewards)
        dones = np.array(dones)
        donesMask = (dones==False).astype(np.float32).reshape(-1)
        dd = torch.tensor(donesMask).to(self.device)
        donesMask = torch.unsqueeze(dd, dim=1)
  
        with torch.no_grad():
            nAction, logProb, _, entropy, _ = \
                self.agent.forward(nstates, lstmState=nlstmState)
            c1, c2 = self.agent.criticForward(nstates, nAction, lstmState=(nlstmState[1], nlstmState[2]))
            minc = torch.min(c1, c2).detach()
        gT = self.getReturn(rewards, dones, minc)
        gT -= self.tempValue * logProb * donesMask
        
        if self.fixedTemp:
            self.zeroGrad()
            lossC1, lossC2 = self.agent.calQLoss(
                states.detach(),
                gT.detach(),
                actions.detach(),
                (lstmState[1], lstmState[2])
            )
            lossC1.backward()
            lossC2.backward()

            self.cFOptim01.step()
            self.cFOptim02.step()
            self.cOptim01.step()
            self.cOptim02.step()

            lossP, lossT = self.agent.calALoss(
                states.detach(),
                lstmState,
                alpha=self.tempValue
            )
            lossP.backward()
            self.aFOptim01.step()
            self.aOptim.step()

        normA = calGlobalNorm(self.actor) + calGlobalNorm(self.actorFeature01)
        normC = calGlobalNorm(self.critic01) + calGlobalNorm(self.criticFeature01) 
        norm = normA + normC
        
        entropy = entropy.mean().cpu().detach().numpy()
        lossP = lossP.cpu().sum().detach().numpy()
        lossC1 = lossC1.cpu().sum().detach().numpy()
        lossC2 = lossC2.cpu().sum().detach().numpy()
        loss = (lossC1 + lossC2)/2 + lossP + lossT

        if self.writeTMode:
            self.writer.add_scalar('Action Gradient Mag', normA, step)
            self.writer.add_scalar('Critic Gradient Mag', normC, step)
            self.writer.add_scalar('Gradient Mag', norm, step)
            self.writer.add_scalar('Entropy', entropy, step)
            self.writer.add_scalar('Loss', loss, step)
            self.writer.add_scalar('Policy Loss', lossP, step)
            self.writer.add_scalar('Critic Loss', (lossC1+lossC2)/2, step)
        self.replayMemory.clear()

    def getReturn(self, reward, done, minC):
        """
        input:
            reward:[np.array]  
                shape:[step, nAgent]
            done:[np.array]
                shape:[step, nAgent]
            minC:[tensor]
                shape[step*nAgent, 1]
            step:int
        """
        nAgent = self.nAgent
        gT = []
        step = len(reward)
        for i in range(nAgent):
            rewardAgent = reward[:, i]  # [step]
            doneAgent = done[:, i]  # [step]
            minCAgent = minC[i * step:(i+1)*step]
            GT = []

            ind = np.where(doneAgent == True)[0]
            div = len(ind) + 1
            if div == 1:
                if doneAgent[-1]:
                    tempGT = [torch.tensor([rewardAgent[-1]]).to(self.device).float()]
                else:
                    tempGT = [minCAgent[-1]]
                rewardAgent = rewardAgent[::-1]
                for i in range(len(rewardAgent)-1):
                    tempGT.append(rewardAgent[i+1] + self.gamma*tempGT[i])
                GT = torch.stack(tempGT[::-1], dim=0)
                gT.append(GT)
            else:
                j = 0
                for i in ind:
                    divRAgent = rewardAgent[j:i+1]
                    divDAgent = doneAgent[j:i+1]
                    divCAgent = minCAgent[j:i+1]
                    j = i+1
                    
                    if divDAgent[-1]:
                        tempGT = [torch.tensor([divRAgent[-1]]).to(self.device).float()]
                    else:
                        tempGT = [divCAgent[-1]]
                    divRAgent = divRAgent[::-1]
                    for i in range(len(divRAgent)-1):
                        tempGT.append(divRAgent[i+1] + self.gamma*tempGT[i])
                    GT.append(torch.stack(tempGT[::-1], dim=0))
                divRAgent = rewardAgent[j:]
                
                divDAgent = doneAgent[j:]
                divCAgent = minCAgent[j:]
                if len(divDAgent) != 0:
                    if divDAgent[-1]:
                        tempGT = [torch.tensor([divRAgent[-1]]).to(self.device).float()]
                    else:
                        tempGT = [divCAgent[-1]]
                    divRAgent = divRAgent[::-1]
                    for i in range(len(divRAgent)-1):
                        tempGT.append(divRAgent[i+1] + self.gamma*tempGT[i])
                    GT.append(torch.stack(tempGT[::-1], dim=0))

                gT.append(torch.cat(GT, dim=0))
        return torch.cat(gT, dim=0)    

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
        hCState01 = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)
        cCState01 = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)
        hCState02 = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)
        cCState02 = torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device)

        lstmState = ((hAState, cAState), (hCState01, cCState01), (hCState02, cCState02))
        return lstmState
    
    def setZeroLSTMState(self, lstmState, idx):
        (ha, ca), (hc1, cc1), (hc2, cc2) = lstmState

        hAState = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        cAState = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        hCState01 = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        cCState01 = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        hCState02 = torch.zeros(1, 1, self.hiddenSize).to(self.device)
        cCState02 = torch.zeros(1, 1, self.hiddenSize).to(self.device)

        ha[:, idx:idx+1, :] = hAState
        ca[:, idx:idx+1, :] = cAState
        hc1[:, idx:idx+1, :] = hCState01
        cc1[:, idx:idx+1, :] = cCState01
        hc2[:, idx:idx+1, :] = hCState02
        cc2[:, idx:idx+1, :] = cCState02

        lstmState = ((ha, ca), (hc1, cc1), (hc2, cc2))

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
        stateT = torch.cat(stateT, dim=0)
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
            nStateT = torch.cat(nStateT, dim=0).to(self.device)
            nAction, nnlstmState = self.getAction(nStateT, lstmState=nlstmState)
            self.appendMemory(
                ((stateT, lstmState), action.copy(),
                    reward*self.rScaling, (nStateT, nlstmState), done.copy())
            )
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
            if step % self.updateStep == 0:
                self.train(step)
            
            if step % 200 == 0:
                episodeReward = np.array(episodeReward)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:3f}  
                """.format(step, reward))
                episodeReward = []

                torch.save(self.agent.state_dict(), self.sPath)