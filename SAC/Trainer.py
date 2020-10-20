import cv2
import torch
import random
import numpy as np

from baseline.baseTrainer import OFFPolicy
from SAC.Agent import sacAgent
from baseline.utils import getOptim, calGlobalNorm
from collections import deque


class sacTrainer(OFFPolicy):

    def __init__(self, fName):
        super(sacTrainer, self).__init__(fName)
        self.agent = sacAgent(self.aData)
        self.tAgent = sacAgent(self.aData)
        if self.lPath != "None":
            self.agent.load_state_dict(
                torch.load(self.lPath, map_location=self.device)
            )
        if 'fixedTemp' in self.keyList:
            self.fixedTemp = self.data['fixedTemp'] == "True"
            if 'tempValue' in self.keyList:
                self.tempValue = self.data['tempValue']
        else:
            self.fixedTemp = False
            self.tempValue = self.agent.temperature.exp()

        self.device = torch.device(self.device)
        self.tAgent.load_state_dict(self.agent.state_dict())

        self.obsSets = []
        for i in range(self.nAgent):
            for j in range(self.sSize[0]):
                self.obsSets.append(deque(maxlen=self.sSize[0]))
    
        self.initializePolicy()
        self.replayMemory = deque(maxlen=self.nReplayMemory)
        self.sPath += self.data['envName']+'_'+str(self.bSize) + \
            '_'+str(~self.fixedTemp)
        if self.fixedTemp:
            self.sPath += '_'+str(int(self.tempValue*100))+'.pth'
        else:
            self.sPath += '.pth'
        
        if self.writeTMode:
            self.writeTrainInfo()
    
    def writeTrainInfo(self):
        super(sacTrainer, self).writeTrainInfo()
        
        if self.sMode:
            self.info += """
        sMode : True 
        tau : {:3f} 
            """.format(self.tau)
        else:
            self.info += """
        sMode : False
            """
        
        if self.fixedTemp:
            self.info += """
        fixedTemp : True
        tempValue : {}
            """.format(self.tempValue)
        else:
            self.info += """
        fixedTemp : False
        """

        print(self.info)

        self.writer.add_text('info', self.info, 0)

    def reset(self):
        for i in range(self.nAgent):
            for j in range(self.sSize[0]):
                self.obsSets[i].append(np.zeros(self.sSize[1:]))
    
    def ppState(self, obs, id=0):
        state = np.zeros(self.sSize)

        if state.ndim > 2:
            obs = cv2.resize(obs, (self.sSize[1:]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(
                obs, 
                (self.sSize[1:])
                )
        self.obsSets[id].append(obs)
        
        for i in range(self.sSize[0]):
            state[i] = self.obsSets[id][i]
        
        if state.ndim > 2:
            state = np.uint8(state)
        return state
    
    def ppEvalState(self, obs):
        state = np.zeros(self.sSize)

        if state.ndim > 2:
            obs = cv2.resize(obs, (self.sSize[1:]))
            obs = np.uint8(obs)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(
                obs, 
                (self.sSize[1:])
                )
        self.evalObsSet.append(obs)
        
        for i in range(self.sSize[0]):
            state[i] = self.evalObsSet[i]
        
        if state.ndim > 2:
            state = np.uint8(state)
        return state

    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.actor, self.policy, self.critic01, self.critic02 = \
            self.agent.actor, self.agent.policy, self.agent.critic01, self.agent.critic02
        self.tCritic01, self.tCritic02 = \
            self.tAgent.critic01, self.tAgent.critic02
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.pOptim = getOptim(self.optimData[optimKey], self.policy)
            if optimKey == 'critic':
                self.cOptim1 = getOptim(self.optimData[optimKey], self.critic01)
                self.cOptim2 = getOptim(self.optimData[optimKey], self.critic02)
            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(self.optimData[optimKey], self.tempValue)
                 
    def getAction(self, state, dMode=False):
        if torch.is_tensor(state) is False:
            state = torch.tensor(state).to(self.device).float()
        
        with torch.no_grad():
            if dMode:
                action = torch.tanh(self.actor.forward(state))
            else:
                action, logProb, critics, _ = self.agent.forward(state)

        return action[0].cpu().detach().numpy()
    
    def targetNetUpdate(self):
        if self.sMode:
            with torch.no_grad():
                for tC1, tC2, C1, C2 in zip(
                    self.tCritic01.parameters(), 
                    self.tCritic02.parameters(), 
                    self.critic01.parameters(), 
                    self.critic02.parameters()):
                    temp1 = self.tau * C1 + (1 - self.tau) * tC1
                    temp2 = self.tau * C2 + (1 - self.tau) * tC2

                    tC1.copy_(temp1)
                    tC2.copy_(temp2)

    def appendMemory(self, data):
        return self.replayMemory.append(data)
    
    def zeroGrad(self):
        self.cOptim1.zero_grad()
        self.cOptim2.zero_grad()
        self.aOptim.zero_grad()
        self.policy.zero_grad()
        if self.fixedTemp is False:
            self.tOptim.zero_grad()
    
    def eval(self, step):
        episodeReward = []
        for i in range(100):
            obs = self.evalEnv.reset()
            for j in range(self.sSize[0]):
                self.evalObsSet.append(np.zeros(self.sSize[1:]))
            state = self.ppEvalState(obs)
            action = self.getAction(state, dMode=True)
            done = False
            rewardT = 0

            while done is False:
                obs, reward, done, _ = self.evalEnv.step(action)
                nState = self.ppEvalState(obs)
                action = self.getAction(nState, dMode=True)
                rewardT += reward
                if done:
                    episodeReward.append(rewardT)
        episodeReward = np.array(episodeReward).mean()
        print("""
        The performance of Deterministic policy is {:3f}
        """.format(episodeReward))
        if self.writeTMode:
            self.writer.add_scalar("Eval Reward", episodeReward, step)
        if episodeReward > self.best:
            self.best = episodeReward
            path = self.data['sPath'] + self.data['envName']+'_'+str(self.best)+'.pth'
            torch.save(self.agent.state_dict(), path)

    def train(self, step):
        miniBatch = random.sample(self.replayMemory, self.bSize)
        
        states, actions, rewards, nStates, dones = \
            [], [], [], [], []
        
        for i in range(self.bSize):
            states.append(miniBatch[i][0])
            actions.append(miniBatch[i][1])
            rewards.append(miniBatch[i][2])
            nStates.append(miniBatch[i][3])
            dones.append(miniBatch[i][4])

        actionsT = torch.tensor(actions).to(self.device).float()
        nStatesT = torch.tensor(nStates).to(self.device).float()
        statesT = torch.tensor(states).to(self.device).float()
        with torch.no_grad():
            nActionsT, logProb, __, entropy = \
                self.agent.forward(nStatesT)
            nStatesT = nStatesT.view((nStatesT.shape[0], -1))
            nStateAction = torch.cat((nStatesT, nActionsT), dim=1)
            target1, target2 = self.tCritic01(nStateAction), self.tCritic02(nStateAction)

        for i in range(self.bSize):
            if dones[i]:
                target1[i] = rewards[i]
                target2[i] = rewards[i]
            else:
                target1[i] = \
                    rewards[i] + self.gamma * (target1[i] - self.tempValue * logProb[i])

                target2[i] = \
                    rewards[i] + self.gamma * (target2[i] - self.tempValue * logProb[i])
        self.agent.train()
        self.tAgent.eval()
        if self.fixedTemp:
            lossC1, lossC2, lossP, lossT = self.agent.calLoss(
                statesT, 
                (target1.detach(), target2.detach()),
                actionsT,
                alpha=self.tempValue
                )
        else:
            lossC1, lossC2, lossP, lossT = self.agent.calLoss(
                statesT.detach(s), 
                (target1.detach(), target2.detach()),
                actionsT.detach()
                )
        
        self.zeroGrad()
        lossP.backward()
        self.critic01.zero_grad()
        self.critic02.zero_grad()

        lossC1.backward()
        lossC2.backward()

        if self.fixedTemp is False:
            lossT.backward()
            self.tOptim.step()
        
        self.cOptim1.step()
        self.cOptim2.step()
        self.aOptim.step()
        self.pOptim.step()
        
        normA = calGlobalNorm(self.actor)
        normC1 = calGlobalNorm(self.critic01)
        normC2 = calGlobalNorm(self.critic02)
        normP = calGlobalNorm(self.policy)

        norm = normA + normC1 + normC2 + normP
        entropy = entropy.mean().cpu().detach().numpy()
        lossP = lossP.cpu().sum().detach().numpy()
        lossC1 = lossC1.cpu().sum().detach().numpy()
        lossC2 = lossC2.cpu().sum().detach().numpy()
        lossT = lossT.cpu().sum().detach().numpy()
        loss = (lossC1 + lossC2)/2 + lossP + lossT

        if self.writeTMode:
            self.writer.add_scalar('Action Gradient Mag', normA, step)
            self.writer.add_scalar('Critic1 Gradient Mag', normC1, step)
            self.writer.add_scalar('Critic2 Gradient Mag', normC2, step)
            self.writer.add_scalar('Gradient Mag', norm, step)
            self.writer.add_scalar('Entropy', entropy, step)
            self.writer.add_scalar('Loss', loss, step)
            self.writer.add_scalar('Policy Loss', lossP, step)
            self.writer.add_scalar('Critic Loss', (lossC1+lossC2)/2, step)
            if self.fixedTemp is False:
                self.writer.add_scalar('Temp Loss', lossT, step)

        return loss, entropy

    def run(self):
        step = 0
        episode = 0
        Loss = []
        episodicReward = []
        
        while 1:
            self.reset()

            if self.uMode:
                envInfo = self.env.reset(train_mode=~self.inferMode)[self.brain]
                obs = envInfo.vector_observations[0]
            else:
                obs = [self.env.reset()]
            stateT = []
            action = []
            for b in range(self.nAgent):
                ob = obs[b]
                state = self.ppState(ob, id=b)
                action.append(self.getAction(state))
                stateT.append(state)
            
            doneT = False
            dones = [False for i in range(self.nAgent)]
            episodeReward = 0

            while doneT is False:
                nState = []
                rewards = []

                if self.uMode:
                    envInfo = self.env.step(action)[self.brain]
                    obs = envInfo.vector_observations
                    rewards = envInfo.rewards
                    donesN = envInfo.dones
                    for b in range(self.nAgent):
                        ob = obs[b]
                        state = self.ppState(ob, id=b)
                        nState.append(state)
                        if dones[b] is False:
                            self.appendMemory((
                                stateT[b], action[b], 
                                rewards[b]*self.rScaling, nState[b], donesN[b]))
                            episodeReward += rewards[b]
                        action[b] = self.getAction(state)
                    dones = donesN

                else:
                    for b in range(self.nAgent):
                        ob, reward, done, _ = self.env.step(action[b])
                        state = self.ppState(ob, id=b)
                        nState.append(state)
                        episodeReward += reward
                        dones[b] = done
                        self.appendMemory((
                            stateT[b], action[b],
                            reward*self.rScaling, nState[b], dones[b]))
                        action[b] = self.getAction(nState[b])
                step += 1
                donesFloat = np.array(dones, dtype=np.float32)
                if np.sum(donesFloat) == self.nAgent:
                    doneT = True
                
                if step >= self.startStep and self.inferMode is False:
                    loss, entropy =\
                        self.train(step)
                    Loss.append(loss)
                
                stateT = nState
                if self.renderMode and self.uMode is False:
                    self.env.render()
                
                if step > self.startStep:
                    self.targetNetUpdate()
                
                if step % self.evalP == 0 and step > self.startStep and self.uMode is False:
                    self.eval(step)

                if doneT and step > self.startStep:
                    episode += 1
                    episodicReward.append(episodeReward)
                    if self.writeTMode:
                        self.writer.add_scalar('Reward', episodeReward, step)

                    if self.fixedTemp:
                        alpha = self.tempValue
                    else:
                        alpha = self.tempValue.cpu().detach().numpy()
                    
                    if episode % self.episodeP == 0:
                        Loss = np.array(Loss).mean()
                        episodicReward = np.array(episodicReward).mean()
                        
                        print("""
                        Episode : {:4d} // Step : {:5d} // Loss : {:3f}
                        Reward : {:3f}  // alpha: {:3f}
                        """.format(episode, step, Loss, episodicReward, alpha))
                        Loss = []
                        episodicReward = []
                    
                    if step > self.startStep and episode % 10 == 0 and self.inferMode is False:

                        torch.save(self.agent.state_dict(), self.sPath)