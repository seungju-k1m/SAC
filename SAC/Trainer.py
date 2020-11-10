import cv2
import torch
import random
import math
import numpy as np
from baseline.baseTrainer import OFFPolicy, ONPolicy
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

        else:
            self.devic = torch.device("cpu")
        self.tAgent.load_state_dict(self.agent.state_dict())
        if 'gpuOverload' in self.data.keys():
            self.gpuOverload = self.data['gpuOverload'] == 'True'
        else:
            self.gpuOverload = False

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
        rState, targetOn, lidarPt = obs[:6], obs[6], obs[7:]
        targetPos = np.reshape(rState[:2], (1, 2))
        if self.gpuOverload:
            rState = torch.tensor(rState)
            lidarImg = torch.zeros(self.sSize).to(self.device)
        else:
            lidarImg = np.zeros(self.sSize)
        lidarPt = lidarPt[lidarPt != 0]
        lidarPt -= 1000
        lidarPt = np.reshape(lidarPt, (-1, 2))
        R = [[math.cos(rState[-1]), -math.sin(rState[-1])], [math.sin(rState[-1]), math.cos(rState[-1])]]
        R = np.array(R)
        lidarPt = np.dot(lidarPt, R)
        for pt in lidarPt:
            
            locX = int(((pt[0]+7) / 14)*self.sSize[-1])
            locY = int(((pt[1]+7) / 14)*self.sSize[-1])

            if locX == self.sSize[-1]:
                locX -= 1
            if locY == self.sSize[-1]:
                locY -= 1

            lidarImg[0, locY, locX] = 1.0
        if targetOn == 1:
            pt = np.dot(targetPos, R)[0]
            locX = int(((pt[0]+7) / 14)*self.sSize[-1])
            locY = int(((pt[1]+7) / 14)*self.sSize[-1])
            if locX == self.sSize[-1]:
                locX -= 1
            if locY == self.sSize[-1]:
                locY -= 1
            lidarImg[0, locY, locX] = 10
            # showLidarImg(lidarImg)
        # lidarImg = lidarImg.type(torch.uint8)
        # if (id % 100 == 0):
        #     showLidarImg(lidarImg)
        if self.gpuOverload:
            lidarImg = lidarImg.type(torch.uint8)
        else:
            lidarImg = np.uint8(lidarImg)

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
        
        if self.gpuOverload:
            nRStatesT = torch.stack(nRStates, 0).to(self.device).float()
            rStatesT = torch.stack(rStates, 0).to(self.device).float()
            lStatesT = torch.stack(lStates, 0).to(self.device).float()
            nLStatesT = torch.stack(nLStates, 0).to(self.device).float()
        else:
            nRStatesT = torch.tensor(nRStates).to(self.device).float()
            rStatesT = torch.tensor(rStates).to(self.device).float()
            lStatesT = torch.tensor(lStates).to(self.device).float()
            nLStatesT = torch.tensor(nLStates).to(self.device).float()
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
        obsState = np.zeros((self.nAgent, 1447))
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
                state = self.ppState(ob, id=step)
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


def ppState_(obs):
        """
        input:
            obs:[np.array]
                shpae:[1447,]
        output:
            lidarImg:[tensor. device]
                shape[1, 96, 96]
        """
        device = torch.device("cpu")
        sSize = [1, 96, 96]
        rState, targetOn, lidarPt = obs[:6], obs[6], obs[7:]
        targetPos = np.reshape(rState[:2], (1, 2))
        rState = torch.tensor(rState).to(device)
        lidarImg = torch.zeros((1, 96, 96)).to(device)
        lidarPt = lidarPt[lidarPt != 0]
        lidarPt -= 1000
        lidarPt = np.reshape(lidarPt, (-1, 2))
        R = [[math.cos(rState[-1]), -math.sin(rState[-1])], [math.sin(rState[-1]), math.cos(rState[-1])]]
        R = np.array(R)
        lidarPt = np.dot(lidarPt, R)
        for pt in lidarPt:
            
            locX = int(((pt[0]+7) / 14)*sSize[-1]-1)
            locY = int(((pt[1]+7) / 14)*sSize[-1]-1)

            if locX == (sSize[-1]-1):
                locX -= 1
            if locY == (sSize[-1]-1):
                locY -= 1

            lidarImg[0, locY+1, locX+1] = 1.0
        if targetOn == 1:
            pt = np.dot(targetPos, R)[0]
            locX = int(((pt[0]+7) / 14)*sSize[-1])
            locY = int(((pt[1]+7) / 14)*sSize[-1])
            if (locX == sSize[-1]-1):
                locX -= 1
            if (locY == sSize[-1]-1):
                locY -= 1
            lidarImg[0, locY+1, locX+1] = 10
        lidarImg[0, 0, :6] = rState
        
        return lidarImg


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
            self.sPath += str(int(self.tempValue * 100)) +'.pth'
        else:
            self.sPath += '.pth'
        self.sSize = self.aData['sSize']
        self.hiddenSize = self.aData['actorFeature02']['hiddenSize']
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
        rState, targetOn, lidarPt = obs[:6], obs[6], obs[7:720+7]
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
                 
    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        self.actor, self.actorFeature01, self.actorFeature02 = \
            self.agent.actor, self.agent.actorFeature01,  self.agent.actorFeature02
        self.criticFeature01, self.critic01 = \
            self.agent.criticFeature01_1, self.agent.critic01
        self.criticFeature02,  self.critic02 = \
            self.agent.criticFeature01_2, self.agent.critic02

        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.aFOptim01 = getOptim(self.optimData[optimKey], self.actorFeature01)
                self.aFOptim02 = getOptim(self.optimData[optimKey], self.actorFeature02)
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
        self.aFOptim02.zero_grad()

        self.cOptim01.zero_grad()
        self.cFOptim01.zero_grad()

        self.cOptim02.zero_grad()
        self.cFOptim02.zero_grad()

    def getAction(self, state, lstmState=None, dMode=False):
        """
        input:
            state:
                dtype:tensor
                shape:[nAgent, 1, 96, 96]
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
            lstmState = (hAState, cAState)

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

            nstates.append(data[3][0])
            nhA.append(data[3][1][0])
            ncA.append(data[3][1][1])

            dones.append(data[4])
        
        states = torch.cat(states, dim=0).to(self.device)
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
        donesMask = (dones==False).astype(np.float32).reshape(-1)
        dd = torch.tensor(donesMask).to(self.device)
        donesMask = torch.unsqueeze(dd, dim=1)
  
        with torch.no_grad():
            nAction, logProb, _, entropy, _ = \
                self.agent.forward(nstates, lstmState=nlstmState)
            c1, c2 = self.agent.criticForward(nstates, nAction)
            minc = torch.min(c1, c2).detach()
        gT = self.getReturn(rewards, dones, minc)
        gT -= self.tempValue * logProb * donesMask
        
        if self.fixedTemp:
            self.zeroGrad()
            lossC1, lossC2 = self.agent.calQLoss(
                states.detach(),
                gT.detach(),
                actions.detach()
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
            self.aFOptim02.step()
            self.aOptim.step()

        normA = calGlobalNorm(self.actor) + calGlobalNorm(self.actorFeature01) + calGlobalNorm(self.actorFeature02)
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