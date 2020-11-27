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
            self.fixedTemp = self.data['fixedTemp']
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
        self.gpuOverload = self.data['gpuOverload']

        self.obsSets = []
        for i in range(self.nAgent):
            for j in range(self.sSize[0]):
                self.obsSets.append(deque(maxlen=self.sSize[0]))
    
        self.initializePolicy()
        self.replayMemory = deque(maxlen=self.nReplayMemory)
        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        self.sPath += name + '_' + str(self.nAgent) + '_imgMode_'
        if self.fixedTemp:
            self.sPath += str(int(self.tempValue * 100)) + '.pth'
        else:
            self.sPath += '.pth'
        
        if self.fixedTemp:
            self.sPath += \
                str(self.nAgent)+self.data['envName'][-2:]+'_'+str(int(self.tempValue*100))+'.pth'
        else:
            self.sPath += '.pth'
        
        if self.writeTMode:
            self.writeTrainInfo()
    
    def writeTrainInfo(self):
        super(sacTrainer, self).writeTrainInfo()
        key = self.data.keys()
        for k in key:
            if k == 'agent' or k == 'optim':
                pass
            else:
                data = self.data[k]
                self.info += """{}:{}
                """.format(k, data)
        print(self.info)
        self.writer.add_text('info', self.info, 0)   

    def reset(self):
        for i in range(self.nAgent):
            for j in range(self.sSize[0]):
                self.obsSets[i].append(np.zeros(self.sSize[1:]))
    
    def resetInd(self, id=0):
        for j in range(self.sSize[0]):
            self.obsSets[0].append(np.zeros(self.sSize[1:]))
    
    def ppState(self, obs):
        """
        args:
            obs:[np.array]
                observation
                shape:[1447, ]
        
        output:
            rState:[torch.tensor]
                state of robot. each element corresponds to relative pos x, y, 
                velocity, yaw rate and y_angle.
                shape:[6, ]
            lidarImg:[torch.tensor uint8]
                project lidar point to lidar img.
                shape:[1, 96, 96]
        """
        rState, lidarPt = obs[:6], obs[7:7 + self.sSize[-1]]
        if self.gpuOverload:
            rState = torch.tensor(rState).to(self.device)
            lidarImg = torch.tensor(lidarPt).to(self.device).float()
        else:
            rState = torch.tensor(rState)
            lidarImg = torch.tensor(lidarPt).float()
        
        lidarImg = torch.unsqueeze(lidarImg, dim=0)

        return (rState, lidarImg)

    def genOptim(self):
        """
        Generate optimizer of each network.
        """
        optimKeyList = list(self.optimData.keys())
        self.aF = self.agent.aF.to(self.device)
        self.cF1 = self.agent.cF1.to(self.device)
        self.cF2 = self.agent.cF2.to(self.device)
        self.actor = self.agent.actor.to(self.device)
        self.critic01 = self.agent.critic01.to(self.device)
        self.critic02 = self.agent.critic02.to(self.device)

        self.tCF1 = self.tAgent.cF1.to(self.device)
        self.tCF2 = self.tAgent.cF2.to(self.device)
        self.tCritic01 = self.tAgent.critic01.to(self.device)
        self.tCritic02 = self.tAgent.critic02.to(self.device)
        
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], (self.aF, self.actor))
            if optimKey == 'critic':
                self.cOptim1 = getOptim(self.optimData[optimKey], (self.cF1, self.critic01))
                self.cOptim2 = getOptim(self.optimData[optimKey], (self.cF2, self.critic02))
            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(
                        self.optimData[optimKey], [self.tempValue], floatV=True)
                 
    def getAction(self, state, dMode=False):
        """
        sample the action from the actor network!

        args:
            state:[tuple]
                consists of rState and lidarImg
            dMode:[bool]
                In the policy evalution, action is not sampled, then determined by actor.
        
        output:
            action:[np.array]
                action
                shape:[2, ]
        """
        with torch.no_grad():
            if dMode:
                action = self.agent.actorForward(state, dMode=False)
            else:
                action, logProb, critics, _ = self.agent.forward(state)

        return action[0].cpu().detach().numpy()
    
    def targetNetUpdate(self):
        """
        Update the target Network
        """
        with torch.no_grad():
            for tC1, tC2, C1, C2, tcF1, tcF2, cF1, cF2 in zip(
                    self.tCritic01.parameters(), 
                    self.tCritic02.parameters(), 
                    self.critic01.parameters(), 
                    self.critic02.parameters(),
                    self.tCF1.parameters(),
                    self.tCF2.parameters(),
                    self.cF1.parameters(),
                    self.cF2.parameters()):
                temp1 = self.tau * C1 + (1 - self.tau) * tC1
                temp2 = self.tau * C2 + (1 - self.tau) * tC2
                temp3 = self.tau * cF1 + (1 - self.tau) * tcF1
                temp4 = self.tau * cF2 + (1 - self.tau) * tcF2

                tC1.copy_(temp1)
                tC2.copy_(temp2)
                tcF1.copy_(temp3)
                tcF2.copy_(temp4)

    def appendMemory(self, data):
        return self.replayMemory.append(data)
    
    def zeroGrad(self):
        self.cOptim1.zero_grad()
        self.cOptim2.zero_grad()
        self.aOptim.zero_grad()
        if self.fixedTemp is False:
            self.tOptim.zero_grad()
    
    def train(self, step):
        """
        Train the network. it consists of 4 procedure.
        1. preprocess the memory which comes from the replay buffer.
        2. detach the memory from the computer's resource and then get the target value
           for the state-action network update.
        3. calculate the loss, then get the gradient.
        4. Finally, log the statistics of the training, e.g norm, entropy
        """
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

        with torch.no_grad():
            nActionsT, logProb, __, entropy = \
                self.agent.forward((nRStatesT, nLStatesT))
            cs1, cs2 = \
                self.tCF1(nLStatesT), self.tCF2(nLStatesT)
            
            c1, c2 = \
                torch.cat((nActionsT, nRStatesT, cs1), dim=1), torch.cat((nActionsT, nRStatesT, cs2), dim=1)
            
            target1, target2 = \
                self.critic01(c1), self.critic02(c2)
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

        normA = calGlobalNorm(self.actor) + calGlobalNorm(self.aF)
        normC1 = calGlobalNorm(self.critic01) + calGlobalNorm(self.cF1)
        normC2 = calGlobalNorm(self.critic02) + calGlobalNorm(self.cF2)

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
                self.writer.add_scalar('alpha', self.tempValue.exp().detach().cpu().numpy()[0], step)
                
        return loss, entropy
    
    def getObs(self, init=False):
        """
        Get the observation from the unity Environment.
        The environment provides the vector which has the 1447 length.
        As you know, two type of step is provided from the environment.
        """
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
            state = self.ppState(ob)
            action.append(self.getAction(state))
            stateT.append(state)
        action = np.array(action)
        while 1:
            nState = []
            self.checkStep(np.array(action))
            obs, rewards, donesN_ = self.getObs()
            for b in range(self.nAgent):
                ob = obs[b]
                state = self.ppState(ob)
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
                    self.resetInd(id=b)
                    episodicReward.append(episodeReward[b])
                    episodeReward[b] = 0

            if step >= int(self.startStep/self.nAgent) and self.inferMode is False:
                loss, entropy =\
                    self.train(step)
                Loss.append(loss)
                self.targetNetUpdate()

            stateT = nState
            step += 1
            if (step % 400 == 0) and step > int(self.startStep/self.nAgent):
                reward = np.array(episodicReward).mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                if self.fixedTemp:
                    alpha = self.tempValue
                else:
                    alpha = self.tempValue.exp().cpu().detach().numpy()[0]
                
                Loss = np.array(Loss).mean()
                    
                print("""
                Step : {:5d} // Loss : {:.3f}
                Reward : {:.3f}  // alpha: {:.3f}
                """.format(step, Loss, reward, alpha))
                Loss = []
                episodicReward = []
                torch.save(self.agent.state_dict(), self.sPath)