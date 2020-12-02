import torch
import datetime
import numpy as np
from baseline.baseTrainer import ONPolicy
from PPO.Agent import ppoAgent
from baseline.utils import getOptim
from collections import deque


def preprocessBatch(f):
    def wrapper(self, step, k, epoch):
        rstate, lidarpt, action, reward, nrstate, nlidarpt, done = \
            [], [], [], [], [], [], []
        for data in self.replayMemory[k]:
            s, a, r, ns, d = data
            rstate.append(s[0])
            lidarpt.append(s[1])
            action.append(a)
            reward.append(r)
            nrstate.append(ns[0])
            nlidarpt.append(ns[1])
            done.append(d)
        state = tuple([torch.cat(rstate, dim=0), torch.cat(lidarpt, dim=0)])
        nstate = tuple([torch.cat(nrstate, dim=0), torch.cat(nlidarpt, dim=0)])
        action = torch.tensor(action).to(self.device).view((-1, 2))
        reward = np.array(reward)
        done = np.array(done)
        f(self, state, action, reward, nstate, done, step, epoch)
    return wrapper


def preprocessState(f):
    def wrapper(self, obs):
        rState = torch.tensor(obs[:, :6]).float().to(self.device)
        lidarPt = torch.tensor(obs[:, 7:127]).float().to(self.device)
        lidarPt = torch.unsqueeze(lidarPt, dim=1)
        state = f(self, (rState, lidarPt))
        return state
    return wrapper


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
        self.labmda = self.data['lambda']

        self.agent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon)
        self.agent.to(self.device)
        self.oldAgent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon)
        self.oldAgent.to(self.device)
        self.oldAgent.update(self.agent)
        
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

        if self.writeTMode:
            self.writeTrainInfo()
    
    def clear(self):
        for i in self.replayMemory:
            i.clear()

    @preprocessState
    def ppState(self, obs):
        return obs

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

    @preprocessBatch
    def train(
        self, 
        state,
        action,
        reward,
        nstate,
        done,
        step,
        epoch
    ):

        with torch.no_grad():
            critic = self.agent.criticForward(state)
            nCritic = self.agent.criticForward(nstate)

        gT, gAE = self.getReturn(reward, critic, nCritic, done)  # step, nAgent
        
        self.zeroGrad()
        lossC = self.agent.calQLoss(
            state,
            gT.detach(),
        
        )
        lossC.backward()
    
        self.cOptim.step()
        normC = self.agent.critic.calculateNorm().cpu().detach().numpy()
        self.zeroGrad()

        minusObj, entropy = self.agent.calAObj(
            self.oldAgent,
            state,
            action,
            gAE.detach()
        )
        minusObj.backward()
        self.agent.actor.clippingNorm(200)
        self.aOptim.step() 
        
        normA = self.agent.actor.calculateNorm().cpu().detach().numpy()

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
        obsState = np.zeros((self.nAgent, 1447), dtype=np.float32)
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

    def run(self):
        episodeReward = []
        k = 0
        Rewards = np.zeros(self.nAgent)
        
        obs = self.getObs(init=True)
        stateT = self.ppState(obs)
        action = self.getAction(stateT)
        step = 1
        while 1:
            self.checkStep(action)
            obs, reward, done = self.getObs()
            Rewards += reward
            nStateT = self.ppState(obs)
            nAction = self.getAction(nStateT)
            u = 0
            for z in range(self.div):
                uu = u + int(self.nAgent/self.div)
                self.replayMemory[z].append(
                        (stateT[u:uu], action[u:uu].copy(),
                        reward[u:uu]*self.rScaling, nStateT[u:uu],
                        done[u:uu].copy()))
                u = uu
            for i, d in enumerate(done):
                if d:
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0

            action = nAction
            stateT = nStateT
            step += 1
            if step % self.updateStep == 0:
                k += 1
                for epoch in range(self.epoch):
                    for j in range(self.div):
                        self.train(step, j, epoch)
                self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    k = 0
            
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