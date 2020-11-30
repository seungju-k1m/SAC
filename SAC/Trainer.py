import torch
import random
import datetime
import numpy as np

from baseline.baseTrainer import OFFPolicy
from SAC.Agent import sacAgent
from baseline.utils import getOptim
from collections import deque


def preprocessBatch(f):
    def wrapper(self, step):
        miniBatch = random.sample(self.replayMemory, self.bSize)

        state, action, reward, nstate, done = \
            [], [], [], [], []
        
        for i in range(self.bSize):
            state.append(miniBatch[i][0][0])
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            nstate.append(miniBatch[i][3][0])
            done.append(miniBatch[i][4])
        state = tuple([torch.cat(state, dim=0).to(self.device).float()])
        action = torch.tensor(action).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        nstate = tuple([torch.cat(nstate, dim=0).to(self.device).float()])

        loss, entropy = f(self, state, action, reward, nstate, done, step)
        return loss, entropy
    return wrapper


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
        time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.sPath += name + '_' + str(time)+'.pth'
        
        if self.writeTMode:
            self.writeTrainInfo()
    
    def writeDict(self, data, key, n=0):
        tab = ""
        for _ in range(n):
            tab += '\t'
        if type(data) == dict:
            for k in data.keys():
                dK = data[k]
                if type(dK) == dict:
                    self.info +=\
                """
            {}{}:
                """.format(tab, k)
                    self.writeDict(dK, k, n=n+1)
                else:
                    self.info += \
            """
            {}{}:{}
            """.format(tab, k, dK)
        else:
            self.info +=\
            """
            {}:{}
            """.format(key, data)
    
    def writeTrainInfo(self):
        self.info = """
        Configuration for this experiment
        """
        key = self.data.keys()
        for k in key:
            data = self.data[k]
            if type(data) == dict:
                self.info +=\
            """
            {}:
            """.format(k)
                self.writeDict(data, k, n=1)
            else:
                self.writeDict(data, k)

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
        state = torch.tensor(obs[:7 + self.sSize[-1]]).float().to(self.device)
        state = torch.unsqueeze(state, dim=0)
        return tuple([state])

    def genOptim(self):
        """
        Generate optimizer of each network.
        """
        optimKeyList = list(self.optimData.keys())
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.agent.actor.buildOptim())
            if optimKey == 'critic':
                self.cOptim1 = getOptim(self.optimData[optimKey], self.agent.critic01.buildOptim())
                self.cOptim2 = getOptim(self.optimData[optimKey], self.agent.critic02.buildOptim())
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

    def appendMemory(self, data):
        return self.replayMemory.append(data)
    
    def zeroGrad(self):
        self.cOptim1.zero_grad()
        self.cOptim2.zero_grad()
        self.aOptim.zero_grad()
        if self.fixedTemp is False:
            self.tOptim.zero_grad()
    
    @preprocessBatch
    def train(
        self,
        state,
        action,
        reward,
        nState,
        done, 
        step
    ):
        with torch.no_grad():
            nActionsT, logProb, __, entropy = \
                self.agent.forward(nState)
            target1, target2 = \
                self.tAgent.criticForward(nState, nActionsT)
            mintarget = torch.min(target1, target2)
            if self.fixedTemp:
                alpha = self.tempValue
            else:
                alpha = self.tempValue.exp()
        for i in range(self.bSize):
            if done[i]:
                mintarget[i] = reward[i]
            else:
                mintarget[i] = \
                    reward[i] + self.gamma * (mintarget[i] - alpha * logProb[i])

        if self.fixedTemp:
            lossC1, lossC2 = self.agent.calQLoss(
                state,
                mintarget.detach(),
                action
            )
            self.zeroGrad()
            lossC1.backward()
            lossC2.backward()
            self.cOptim1.step()
            self.cOptim2.step()

            lossP, lossT = self.agent.calALoss(
                state,
                alpha=self.tempValue)
            
            lossP.backward()
            self.aOptim.step()

        else:
            lossC1, lossC2 = self.agent.calQLoss(
                state,
                mintarget.detach(),
                action
            )
            self.zeroGrad()
            lossC1.backward()
            lossC2.backward()
            self.cOptim1.step()
            self.cOptim2.step()

            lossP, lossT = self.agent.calALoss(
                state
                )
            
            lossP.backward()
            lossT.backward()
            self.aOptim.step()
            self.tOptim.step()

        normA = self.agent.actor.calculateNorm().detach().numpy()
        normC1 = self.agent.critic01.calculateNorm().detach().numpy()
        normC2 = self.agent.critic02.calculateNorm().detach().numpy()

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

    def targetNetUpdate(self):
        self.tAgent.critic01.updateParameter(self.agent.critic01, self.tau)
        self.tAgent.critic02.updateParameter(self.agent.critic02, self.tau)

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
            if (step % 1000 == 0) and step > int(self.startStep/self.nAgent):
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