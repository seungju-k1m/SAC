import torch
import datetime
import numpy as np

from baseline.baseTrainer import ONPolicy
from SAC.Agent import sacAgent
from baseline.utils import getOptim
from collections import deque
from SAC.wrapper import preprocessBatch, preprocessState
from mlagents_envs.base_env import ActionTuple


class sacTrainer(ONPolicy):

    def __init__(self, fName):
        super(sacTrainer, self).__init__(fName)

        # Specify the device for training.

        if self.device != "cpu":
            self.device = torch.device(self.device)
            
        else:
            self.device = torch.device("cpu")
        
        # Specify the Hyper-Parameter
        
        if 'fixedTemp' in self.keyList:
            self.fixedTemp = self.data['fixedTemp']
            if self.fixedTemp:
                if 'tempValue' in self.keyList:
                    self.tempValue = self.data['tempValue']
            else:
                self.fixedTemp = False
                self.tempValue = self.agent.temperature

        self.updateStep = self.data['K1']
        self.K1 = self.data['K1']
        self.K2 = self.data['K2']
        self.epoch = self.data['epoch']
        self.updateOldP = self.data['updateOldP']
        self.replayMemory = deque(maxlen=self.updateStep)
        self.ReplayMemory_Trajectory = deque(maxlen=100000)
        self.LSTMName = self.data['LSTMName']

        # load the agent

        self.agent = sacAgent(self.aData, self.LSTMName)
        self.oldAgent = sacAgent(self.aData, self.LSTMName)
        self.copyAgent = sacAgent(self.aData, self.LSTMName)

        if self.lPath != "None":
            self.agent.load_state_dict(
                torch.load(self.lPath, map_location=self.device)
            )
            self.agent.loadParameters()
        self.agent.to(self.device)

        self.oldAgent.to(self.device)
        self.oldAgent.update(self.agent)

        self.copyAgent.to(self.device)
        self.copyAgent.update(self.agent)
    
        # specify the info of environment and save path.

        pureEnv = self.data['envName'].split('/')
        name = pureEnv[-1]
        time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.sPath += name + '_' + str(time)+'.pth'

        if self.writeTMode:
            self.writeTrainInfo()

    def clear(self):
        self.replayMemory.clear()
    
    @preprocessState
    def ppState(self, obs):
        return tuple([obs])

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

        with torch.no_grad():
            if dMode:
                pass
            else:
                action, logProb, critics, _ = self.oldAgent.forward(state)

        return action[0].cpu().detach().numpy()
    
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
        gT,
        step,
        epoch
    ):

        lossP, lossT = self.agent.calALoss(
            state,
            alpha=self.tempValue)
        lossP.backward()
        if self.fixedTemp:
            lossT.backward()
            self.tOptim.step()
        self.aOptim.step()
        self.zeroGrad()

        lossC1, lossC2 = self.agent.calQLoss(
            state,
            gT.detach(),
            action
        )
        lossC1.backward()
        lossC2.backward()
        self.cOptim1.step()
        self.cOptim2.step()
            
        normA = self.agent.actor.calculateNorm().cpu().detach().numpy()
        normC1 = self.agent.critic01.calculateNorm().cpu().detach().numpy()
        normC2 = self.agent.critic02.calculateNorm().cpu().detach().numpy()

        norm = normA + normC1 + normC2
        lossP = lossP.cpu().sum().detach().numpy()
        lossC1 = lossC1.cpu().sum().detach().numpy()
        lossC2 = lossC2.cpu().sum().detach().numpy()
        lossT = lossT.cpu().sum().detach().numpy()
        loss = (lossC1 + lossC2)/2 + lossP + lossT

        if self.writeTMode:
            with torch.no_grad():
                self.writer.add_scalar('Action Gradient Mag', normA, step)
                self.writer.add_scalar('Critic1 Gradient Mag', normC1, step)
                self.writer.add_scalar('Critic2 Gradient Mag', normC2, step)
                self.writer.add_scalar('Gradient Mag', norm, step)
                self.writer.add_scalar('Loss', loss, step)
                self.writer.add_scalar('Policy Loss', lossP, step)
                self.writer.add_scalar('Critic Loss', (lossC1+lossC2)/2, step)
                self.writer.add_scalar('gT', gT.mean().detach().cpu().numpy()[0], step)
                if self.fixedTemp is False:
                    self.writer.add_scalar('Temp Loss', lossT, step)
                    self.writer.add_scalar(
                        'alpha',
                        self.tempValue.exp().detach().cpu().numpy()[0], step)

    def getObs(self, init=False):
        """
        Get the observation from the unity Environment.
        The environment provides the vector which has the 1447 length.
        As you know, two type of step is provided from the environment.
        """
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        image = decisionStep.obs[0]
        obs = decisionStep.obs[1]
        rewards = decisionStep.reward
        obs = obs.tolist()

        obs = list(map(lambda x: np.array(x), obs))
        obs = np.array(obs)

        done = []
        
        for done_idx in obs[:, -1]:
            done.append(done_idx == 1)
        reward = rewards
        
        obsState = (obs, image)

        if init:
            return obsState
        else:
            return(obsState, reward, done)
    
    def checkStep(self, action):
        decisionStep, terminalStep = self.env.get_steps(self.behaviorNames)
        agentId = decisionStep.agent_id
        act = ActionTuple(
            continuous=action
        )
        if len(agentId) != 0:
            self.env.set_actions(self.behaviorNames, act)
        self.env.step()

    def run(self):
        self.loadUnityEnv()
        episodeReward = []
        k = 0
        Rewards = np.zeros(self.nAgent)
        obs = self.getObs(init=True)
        stateT = self.ppState(obs)
        action = self.getAction(stateT)
        step = 0

        while 1:
            self.checkStep(action)

            obs, reward, done = self.getObs()

            Rewards += reward
            
            nStateT = self.ppState(obs)
            nAction = self.getAction(nStateT)

            with torch.no_grad():
                if self.inferMode is False:
                    self.replayMemory.append(
                        (stateT, 
                         action.copy(),
                         reward * self.rScaling,
                         nStateT,
                         done.copy())
                    )
                    stateT_cpu = tuple([x.cpu() for x in stateT])
                    self.ReplayMemory_Trajectory.append(
                        stateT_cpu
                    )
            
            action = nAction
            stateT = nStateT
            step += 1

            if (step) % (self.updateStep) == 0 and self.inferMode is False:
                k += 1
                self.train(step, self.epoch)
                self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    self.copyAgent.update(self.agent)
                    k = 0
            
            if True in done:
                self.agent.actor.zeroCellState()
                self.agent.critic01.zeroCellState()
                self.agent.critic02.zeroCellState()

                self.oldAgent.actor.zeroCellState()
                self.oldAgent.critic01.zeroCellState()
                self.oldAgent.critic02.zeroCellState()

                self.copyAgent.actor.zeroCellState()
                self.copyAgent.critic01.zeroCellState()
                self.copyAgent.critic02.zeroCellState()

                obs = self.getObs(init=True)
                stateT = self.ppState(obs)
                action = self.getAction(stateT)
                self.checkStep(action)
            
            if step % 3200 == 0:
                episodeReward = np.array(Rewards)
                reward = episodeReward.mean()
                if self.writeTMode:
                    self.writer.add_scalar('Reward', reward, step)

                print("""
                Step : {:5d} // Reward : {:.3f}  
                """.format(step, reward))
                Rewards = np.zeros(self.nAgent)
                if (reward > self.RecordScore):
                    self.RecordScore = reward
                    sPath = './save/PPO/'+self.data['envName']+str(self.RecordScore)+'.pth'
                    torch.save(self.agent.state_dict(), sPath)
                torch.save(self.agent.state_dict(), self.sPath)
