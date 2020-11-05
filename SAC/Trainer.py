import cv2
import torch
import random
import numpy as np

from baseline.baseTrainer import OFFPolicy
from SAC.Agent import sacAgent
from baseline.utils import getOptim, calGlobalNorm, showLidarImg
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
            if self.fixedTemp:
                if 'tempValue' in self.keyList:
                    self.tempValue = self.data['tempValue']
            else:
                self.fixedTemp = False
                self.tempValue = self.agent.temperature

        if self.device != "cpu":
            a = self.device
            self.device = torch.device(self.device)
            torch.cuda.set_device(int(a[-1]))
        else:
            self.devic = torch.device("cpu")
        self.tAgent.load_state_dict(self.agent.state_dict())

        self.obsSets = []
        for i in range(self.nAgent):
            for j in range(self.sSize[0]):
                self.obsSets.append(deque(maxlen=self.sSize[0]))
    
        self.initializePolicy()
        self.replayMemory = deque(maxlen=self.nReplayMemory)
        self.sPath += str(self.bSize) + \
            '_'+str(~self.fixedTemp)
        
        if self.fixedTemp:
            self.sPath += str(self.nAgent)+self.data['envName'][-2:]+'_'+str(int(self.tempValue*100))+'.pth'
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
    
    def resetInd(self, id=0):
        for j in range(self.sSize[0]):
            self.obsSets[0].append(np.zeros(self.sSize[1:]))
    
    def ppState(self, obs, id=0):
        rState, lidarPt = obs[:6], obs[6:]
        rState = torch.tensor(rState).to(self.device)
        lidarPt = lidarPt[lidarPt != 0]
        lidarPt -= 1000
        lidarPt = np.reshape(lidarPt, (-1, 2))
        lidarImg = torch.zeros(self.sSize)
        for pt in lidarPt:
            locX = int(((pt[0]+7) / 14)*self.sSize[-1])
            locY = int(((pt[1]+7) / 14)*self.sSize[-1])

            lidarImg[0, locY, locX] = 1.0
        lidarImg = lidarImg.to(self.device)
        # showLidarImg(lidarImg)

        return (rState, lidarImg)
    
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
        self.actor, self.actorFeature, self.critic01, self.critic02 = \
            self.agent.actor, self.agent.actorFeature,  self.agent.critic01, self.agent.critic02
        self.tCritic01, self.tCritic02 = \
            self.tAgent.critic01, self.tAgent.critic02
        self.tCriticFeature01, self.tCriticFeature02 = \
            self.tAgent.criticFeature01, self.tAgent.criticFeature02
        self.criticFeature01, self.criticFeature02 = \
            self.agent.criticFeature01, self.agent.criticFeature02

        self.actor = self.actor.to(self.device)
        self.critic01, self.critic02 = self.critic01.to(self.device), self.critic02.to(self.device)
        self.tCritic01, self.tCritic02 = \
            self.tCritic01.to(self.device), self.tCritic02.to(self.device)
        self.actorFeature = self.actorFeature.to(self.device)
        self.criticFeature01, self.criticFeature02 = \
            self.criticFeature01.to(self.device), self.criticFeature02.to(self.device)
        self.tCritic01Feature01, self.tCritic01Feature02 = \
            self.tCriticFeature01.to(self.device), self.tCriticFeature02.to(self.device)

        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.aFOptim = getOptim(self.optimData[optimKey], self.actorFeature)
            if optimKey == 'critic':
                self.cOptim1 = getOptim(self.optimData[optimKey], self.critic01)
                self.cFOptim1 = getOptim(self.optimData[optimKey], self.criticFeature01)
                self.cOptim2 = getOptim(self.optimData[optimKey], self.critic02)
                self.cFOptim2 = getOptim(self.optimData[optimKey], self.criticFeature02)
            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(self.optimData[optimKey], [self.tempValue], floatV=True)
                 
    def getAction(self, state, dMode=False):
        
        with torch.no_grad():
            if dMode:
                action = torch.tanh(self.actor(state)[:, :self.aSize])
            else:
                action, logProb, critics, _ = self.agent.forward(state)

        return action[0].cpu().detach().numpy()
    
    def targetNetUpdate(self):
        if self.sMode:
            with torch.no_grad():
                for tC1, tC2, C1, C2, tFC1, tFC2, FC1, FC2 in zip(
                        self.tCritic01.parameters(), 
                        self.tCritic02.parameters(), 
                        self.critic01.parameters(), 
                        self.critic02.parameters(),
                        self.tCriticFeature01.parameters(),
                        self.tCriticFeature02.parameters(),
                        self.criticFeature01.parameters(),
                        self.criticFeature02.parameters()):
                    temp1 = self.tau * C1 + (1 - self.tau) * tC1
                    temp2 = self.tau * C2 + (1 - self.tau) * tC2
                    temp3 = self.tau * FC1 + (1 - self.tau) * tFC1
                    temp4 = self.tau * FC2 + (1 - self.tau) * tFC2

                    tC1.copy_(temp1)
                    tC2.copy_(temp2)
                    tFC1.copy_(temp3)
                    tFC2.copy_(temp4)

    def appendMemory(self, data):
        return self.replayMemory.append(data)
    
    def zeroGrad(self):
        self.cOptim1.zero_grad()
        self.cFOptim1.zero_grad()
        self.cOptim2.zero_grad()
        self.cFOptim2.zero_grad()
        self.aOptim.zero_grad()
        self.aFOptim.zero_grad()
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
        
        rStates, lStates, actions, rewards, nRStates, nLStates, dones = \
            [], [], [], [], [], [], []
        
        for i in range(self.bSize):
            rStates.append(miniBatch[i][0][0])
            lStates.append(miniBatch[i][0][1])
            actions.append(miniBatch[i][1])
            rewards.append(miniBatch[i][2])
            nRStates.append(miniBatch[i][3][0])
            nLStates.append(miniBatch[i][3][1])
            dones.append(miniBatch[i][4])

        nRStatesT = torch.stack(nRStates, 0).to(self.device).float()
        rStatesT = torch.stack(rStates, 0).to(self.device).float()
        lStatesT = torch.stack(lStates, 0).to(self.device).float()
        nLStatesT = torch.stack(nLStates, 0).to(self.device).float()
        actionsT = torch.tensor(actions).to(self.device).float()
        # nRStatesT = torch.tensor(nRStates).to(self.device).float()
        # nLStatesT = torch.tensor(nLStates).to(self.device).float()
        # rStatesT = torch.tensor(rStates).to(self.device).float()
        # lStatesT = torch.tensor(lStates).to(self.device).float()

        with torch.no_grad():
            nActionsT, logProb, __, entropy = \
                self.agent.forward((nRStatesT, nLStatesT))
            
            tCT01 = self.tCriticFeature01(nLStatesT)
            tCT02 = self.tCriticFeature02(nLStatesT)

            cat1 = torch.cat((nActionsT, nRStatesT, tCT01), dim=1)
            cat2 = torch.cat((nActionsT, nRStatesT, tCT02), dim=1)

            target1, target2 = \
                self.tCritic01.forward(cat1), self.tCritic02.forward(cat2)
            mintarget = torch.min(target1, target2)
            if self.fixedTemp:
                alpha = self.tempValue
            else:
                alpha = self.tempValue.exp()
        for i in range(self.bSize):
            if dones[i]:
                mintarget[i] = rewards[i]
            else:
                mintarget[i] = \
                    rewards[i] + self.gamma * (mintarget[i] - alpha * logProb[i])

        if self.fixedTemp:
            lossC1, lossC2 = self.agent.calQLoss(
                (rStatesT.detach(), lStatesT.detach()),
                mintarget.detach(),
                actionsT
            )
            self.zeroGrad()
            lossC1.backward()
            lossC2.backward()
            self.cOptim1.step()
            self.cOptim2.step()

            lossP, lossT = self.agent.calALoss(
                (rStatesT.detach(), lStatesT.detach()),
                alpha=self.tempValue)
            
            lossP.backward()
            self.aOptim.step()

        else:
            lossC1, lossC2 = self.agent.calQLoss(
                (rStatesT.detach(), lStatesT.detach()),
                mintarget.detach(),
                actionsT
            )
            self.zeroGrad()
            lossC1.backward()
            lossC2.backward()
            self.cOptim1.step()
            self.cOptim2.step()

            lossP, lossT = self.agent.calALoss(
                (rStatesT.detach(), lStatesT.detach())
                )
            
            lossP.backward()
            lossT.backward()
            self.aOptim.step()
            self.tOptim.step()

        normA = calGlobalNorm(self.actor)
        normC1 = calGlobalNorm(self.critic01)
        normC2 = calGlobalNorm(self.critic02)

        norm = normA + normC1 + normC2
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
    
    def getObs(self, init=False):
        obsState = np.zeros((self.nAgent, 1446))
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
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        agentId = decisionStep.agent_id
        value = True
        if len(agentId) != 0:
            self.env.set_actions(self.behaviorNames, action)
        else:
            value = False
        self.env.step()
        return value

    def run(self):
        step = 0
        Loss = []
        episodicReward = []
        episodeReward = []

        for i in range(self.nAgent):
            episodeReward.append(0)

        self.reset()
        obs = self.getObs(init=True)
        stateT = []
        action = []
        for b in range(self.nAgent):
            ob = obs[b]
            state = self.ppState(ob, id=b)
            action.append(self.getAction(state))
            stateT.append(state)
        action = np.array(action)

        while 1:
            nState = []
            self.checkStep(np.array(action))
            obs, rewards, donesN_ = self.getObs()
            
            for b in range(self.nAgent):
                ob = obs[b]
                state = self.ppState(ob, id=b)
                nState.append(state)
                self.appendMemory((
                    stateT[b], action[b].copy(), 
                    rewards[b]*self.rScaling, nState[b], donesN_[b]))
                episodeReward[b] += rewards[b]
                if self.inferMode:
                    action[b] = self.getAction(state, dMode=True)
                else:
                    action[b] = self.getAction(state)

                if donesN_[b]:
                    # showLidarImg(state[1])
                    self.resetInd(id=b)
                    episodicReward.append(episodeReward[b])
                    episodeReward[b] = 0

                if step >= self.startStep and self.inferMode is False:
                    loss, entropy =\
                        self.train(step)
                    Loss.append(loss)
                    self.targetNetUpdate()

            stateT = nState
            step += self.nAgent
            if (step % 1000 == 0) and step > self.startStep:

                reward = np.array(episodicReward).mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                if self.fixedTemp:
                    alpha = self.tempValue
                else:
                    alpha = self.tempValue.exp().cpu().detach().numpy()[0]
                
                Loss = np.array(Loss).mean()
                    
                print("""
                Step : {:5d} // Loss : {:3f}
                Reward : {:3f}  // alpha: {:3f}
                """.format(step, Loss, reward, alpha))
                Loss = []
                episodicReward = []
                torch.save(self.agent.state_dict(), self.sPath)