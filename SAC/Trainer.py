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
        self.sPath += name + '_' + str(self.nAgent) + '_LSTMV1_'
        if self.fixedTemp:
            self.sPath += str(int(self.tempValue * 100)) + '.pth'
        else:
            self.sPath += '.pth'
        self.sSize = self.aData['sSize']
        self.hiddenSize = self.aData['Feature']['hiddenSize']
        self.updateStep = self.data['updateStep']
        self.genOptim()

    def ppState(self, obs):
        """
        input:
            obs:[np.array]
                shpae:[1447,]
        """

        rState, lidarPt = obs[:6], obs[7:727]
        rState = torch.tensor(rState).view((1, -1))
        lidarPt = torch.tensor(lidarPt).view((1, -1))

        state = torch.cat((rState, lidarPt), dim=1).float().to(self.device)

        return state
 
    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.Feature = self.agent.Feature.to(self.device)
        self.actor = self.agent.actor.to(self.device)
        self.critic01, self.critic02 =\
            self.agent.critic01.to(self.device), self.agent.critic02.to(self.device)
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.fOptim = getOptim(self.optimData[optimKey], self.Feature)

            if optimKey == 'critic':
                self.cOptim01 = getOptim(self.optimData[optimKey], self.critic01)
                self.cOptim02 = getOptim(self.optimData[optimKey], self.critic02)

            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(self.optimData[optimKey], [self.tempValue], floatV=True)

    def zeroGrad(self):
        self.aOptim.zero_grad()
        self.fOptim.zero_grad()

        self.cOptim01.zero_grad()
        self.cOptim02.zero_grad()

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

    def train(self, step):
        """
        this method is for training!!

        before training, shift the device of idx network and data

        """
        states, hA, cA,  actions, rewards, dones = \
            [], [], [], [], [], []
        nstates, nhA, ncA = \
            [], [], []
        for data in self.replayMemory:
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
            nAction, logProb, _, entropy, _ = \
                self.agent.forward(nstates, lstmState=nlstmState)
            c1, c2 = self.agent.criticForward(nstates, nAction)

            # c1a, c2a = torch.abs(c1), torch.abs(c2)

            # ca = torch.cat((c1a, c2a), dim=1)

            # argmin = torch.argmin(ca, dim=1).view(-1, 1)
            # minc = torch.cat((c1a, c2a), dim=1)
            # c = []
            # for z, i in enumerate(argmin):
            #     c.append(minc[z, i])
            # minc = torch.stack(c, dim=0)
            minc = torch.min(c1, c2)
            minc = minc.detach()

        gT = self.getReturn(rewards, dones, minc, logProb)  # step, nAgent
        
        if self.fixedTemp:
            self.zeroGrad()
            lossC1, lossC2 = self.agent.calQLoss(
                states.detach(),
                gT.detach(),
                actions.detach(),
                lstmState
            )
            lossC1.backward(retain_graph=True)
            lossC2.backward()

            self.cOptim01.step()
            self.cOptim02.step()
            self.fOptim.step()

            lossP, lossT = self.agent.calALoss(
                states.detach(),
                lstmState,
                alpha=self.tempValue
            )
            lossP.backward()
            self.aOptim.step()
            self.fOptim.step()
        
        aN = calGlobalNorm(self.actor)
        aF1N = calGlobalNorm(self.Feature)
        normA = aN + aF1N 

        cN = calGlobalNorm(self.critic01)
        normC = cN

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

    def getReturn(self, reward, done, minC, logProb):
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
        minC = minC.view((step, -1))
        logProb = logProb.view((step, -1))
        for i in range(nAgent):
            rA = reward[:, i]  # [step] , 100
            dA = done[:, i]  # [step] , 100
            mCA = minC[:, i]
            lProbA = logProb[:, i]
            GT = []
            if dA[-1]:
                discounted_r = 0
            else:
                discounted_r = mCA[-1]
            for r, is_terminal, lP in zip(reversed(rA), reversed(dA), reversed(lProbA)):
                if is_terminal:
                    discounted_r = 0
                    lP = 0
                discounted_r = r + self.gamma * (discounted_r - self.tempValue * lP)
                GT.append(discounted_r)
            GT = torch.tensor(GT[::-1]).view((-1, 1)).to(self.device)
            gT.append(GT)

        gT = torch.cat(gT, dim=0)
        gT = gT.view(nAgent, -1)
        gT = gT.permute(1, 0).contiguous()
        gT = gT.view((-1, 1))
        return gT        

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

