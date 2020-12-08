import torch
import datetime
import numpy as np
from baseline.baseTrainer import ONPolicy
from PPO.Agent import ppoAgent
from baseline.utils import getOptim
from collections import deque


# def preprocessBatch(f):
#     def wrapper(self, step, epoch):
#         rstate, action, reward, nrstate, done = \
#             [], [], [], [], []
#         for data in self.replayMemory[0]:
#             s, a, r, ns, d = data
#             rstate.append(s)
#             action.append(a)
#             reward.append(r)
#             nrstate.append(ns)
#             done.append(d)
#         state = tuple([torch.cat(rstate, dim=0)])
#         nstate = tuple([torch.cat(nrstate, dim=0)])
#         action = torch.tensor(action).to(self.device).view((-1, 2))
#         reward = np.array(reward)
#         # reward = (reward - np.mean(reward))/(np.std(reward)+1e-5)
#         done = np.array(done)
        
#         for i in range(epoch): 
#             with torch.no_grad():
#                 critic = self.agent.criticForward(state)
#                 nCritic = self.agent.criticForward(nstate)
#             gT, gAE = self.getReturn(reward, critic, nCritic, done)
#             f(self, state, action, gT, gAE, critic, step, i)
#     return wrapper


def preprocessBatch(f):
    def wrapper(self, step, epoch):
        rstate, action, reward, nrstate, done = \
            [], [], [], [], []
        for data in self.replayMemory[0]:
            s, a, r, ns, d = data
            rstate.append(s)
            action.append(a)
            reward.append(r)
            nrstate.append(ns)
            done.append(d)
            # s, <nAgent, obs> 
        
        State = torch.zeros((10, 0, 126)).to(self.device)
        done = np.array(done)
        done = np.transpose(done, (1, 0))
        reward = np.array(reward)
        reward = np.transpose(reward, (1, 0))
        state = torch.cat(rstate, dim=0)  # <updateStep, nAgent,  obs>
        state = state.permute(1, 0, 2).contiguous()  # nAgent, step, obs
        action = np.array(action)
        action = np.transpose(action, (1, 0, 2))
        nstate = torch.cat(nrstate, dim=0)
        nstate = nstate.permute(1, 0, 2).contiguous()
        targetCritic = []
        targetActor = []
        CriticActor = []
        Action = []
        for s, ns, d, r, a in zip(state, nstate, done, reward, action):
            index = np.where(d == True)[0]
            index = list(index)
            index.append(160)
            index = np.array(index)
            j = 0
            for i in index:
                if (i - j) < 10:
                    j = i
                    continue
                ss = s[j:i]
                rr = r[j:i]
                gT, gAE = [], []
                lastns = ns[i-1:i]
                dd = d[i-1]
                with torch.no_grad():
                    if dd:
                        ss = ss.view((-1, 1, 126))
                        ss = tuple([ss])
                        cc = self.agent.critic.forward(ss)[0]
                        ncc = cc[1:]
                        ncc = torch.cat((ncc, torch.zeros(1, 1)), dim=0)
                    else:
                        ss = torch.cat((ss, lastns), dim=0)
                        ss = ss.view((-1, 1, 126))
                        ss = tuple([ss])
                        cc = self.agent.critic.forward(ss)[0]
                        ncc = cc[1:]
                        cc = cc[:-1]
                    tdError = rr[-1] + self.gamma*ncc[-1] - cc[-1]
                    discountedR = ncc[-1]
                    CriticActor.append(cc)
                    for r_, c_, nc_ in zip(
                        reversed(rr), reversed(cc), reversed(ncc)):
                        gT.append(discountedR)
                        gAE.append(tdError)
                        discountedR = r_ + self.gamma * discountedR
                        error = r_ + self.gamma * nc_ - c_
                        tdError = error + self.gamma * self.labmda * tdError
                    
                    gT = torch.tensor(gT[::-1])
                    gAE = torch.tensor(gAE[::-1])
                    
                k = (i-j) % 10
                sss = s[j:i-k]
                sss = sss.view(10, -1, 126)
                Action.append(a[j:i-k])
                State = torch.cat((State, sss), dim=1)
                targetCritic.append(gT[:i-j-k])
                targetActor.append(gAE[:i-j-k])
                if k != 0:
                    ssss = ss[0][-10:]
                    ssss = ssss.view(10, 1, 126)
                    State = torch.cat((State, ssss), dim=1)
                    Action.append(a[i-10:i])
                    targetCritic.append(gT[-10:])
                    targetActor.append(gAE[-10:])
                j = i

        # state = state.permute(1, 2, 0, 3).contiguous()  # <1, nAgent, updateStep, obs>  
        State = tuple([State])
        Action = np.concatenate(Action, axis=0)
        Action = torch.tensor(Action).view((-1, 2)).to(self.device)
        targetCritic = torch.cat(targetCritic, dim=0).to(self.device)
        targetActor = torch.cat(targetActor, dim=0).to(self.device)
        CriticActor = torch.cat(CriticActor, dim=0).to(self.device)

        # reward = (reward - np.mean(reward))/(np.std(reward)+1e-5)

        for i in range(epoch): 

            f(self, State, Action, targetCritic, targetActor, CriticActor, step, i)
    return wrapper


def preprocessState(f):
    def wrapper(self, obs):
        # rState = torch.tensor(obs[:, :6]).float().to(self.device)
        # lidarPt = torch.tensor(obs[:, 7:127]).float().to(self.device)
        # lidarPt = torch.unsqueeze(lidarPt, dim=1)
        # state = torch.tensor(obs[:, :127]).float().to(self.device)
        # state = f(self, (state))
        # state = tuple([state])
        rState = torch.tensor(obs[:, :6]).float().to(self.device)
        lidarPt = torch.tensor(obs[:, 8:8+self.sSize[-1]]).float().to(self.device)
        state = [torch.unsqueeze(torch.cat((rState, lidarPt), dim=1), dim=0)]
        # state = torch.unsqueeze(state, dim=0)
        return state
    return wrapper


class PPOOnPolicyTrainer(ONPolicy):

    def __init__(self, cfg):
        super(PPOOnPolicyTrainer, self).__init__(cfg)
        
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
        initLogStd = torch.tensor(self.data['initLogStd']).to(self.device).float()
        finLogStd = torch.tensor(self.data['finLogStd']).to(self.device).float()
        annealingStep = self.data['annealingStep']
        self.LSTMNum = self.data['LSTMNum']
        self.agent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMNum=self.LSTMNum)
        self.agent.to(self.device)
        if self.lPath != "None":
            self.agent.load_state_dict(
                torch.load(self.lPath, map_location=self.device)
            )
            self.agent.loadParameters()
        self.oldAgent = ppoAgent(
            self.aData,
            coeff=self.entropyCoeff,
            epsilon=self.epsilon,
            initLogStd=initLogStd,
            finLogStd=finLogStd,
            annealingStep=annealingStep,
            LSTMNum=self.LSTMNum)
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
        self.Number_Episode = 0
        self.Number_Sucess = 0

        if self.writeTMode:
            self.writeTrainInfo()
    
    def clear(self):
        for i in self.replayMemory:
            i.clear()

    @preprocessState
    def ppState(self, obs):
        return tuple([obs])

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
        gT,
        gAE,
        critic,
        step,
        epoch
    ):
        self.zeroGrad()
        lossC = self.agent.calQLoss(
            state,
            gT.detach(),
        
        )
        lossC.backward()
        inputD = []
        for a in self.agent.critic.buildOptim():
            inputD += list(a.parameters())
        torch.nn.utils.clip_grad_norm_(inputD, 5)
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
        inputD = []
        for a in self.agent.actor.buildOptim():
            inputD += list(a.parameters())
        torch.nn.utils.clip_grad_norm_(inputD, 5)
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

    def getReturn(self, reward, critic, nCritic, done, Step_Agent=False):
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
        gAE = torch.cat(gAE, dim=0)

        if Step_Agent:
            gT.view(-1, 1)
            gAE.view(-1, 1)
        else:
            gT = gT.view(nAgent, -1)
            gT = gT.permute(1, 0).contiguous()
            gT = gT.view((-1, 1))

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
        
        done = [False for i in range(self.nAgent)]
        reward = [0 for i in range(self.nAgent)]
        obsState = np.array(obs)
        reward = rewards
        
        # for i, state in zip(agentId, obs):
        #     state = np.array(state)
        #     obsState[i] = state
        #     done[i] = False
        #     reward[i] = rewards[k]
        #     k += 1
        k = 0
        for i, state in zip(tAgentId, tobs):
            state = np.array(state)
            obsState[i] = state
            done[i] = True
            self.Number_Episode += 1
            reward[i] = treward[k]
            if (reward[i]>1):
                self.Number_Sucess += 1
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
    
    def LogSucessRate(self, step):
        if self.writeTMode:
            self.writer.add_scalar("Sucess Rate", (self.Number_Sucess/self.Number_Episode), step)
            self.Number_Episode = 0
            self.Number_Sucess = 0

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
                        (stateT[0][u:uu], action[u:uu].copy(),
                         reward[u:uu]*self.rScaling, nStateT[0][u:uu],
                         done[u:uu].copy()))
                u = uu
            for i, d in enumerate(done):
                if d:
                    episodeReward.append(Rewards[i])
                    Rewards[i] = 0
                    self.agent.clear(i)
                    self.oldAgent.clear(i)

            action = nAction
            stateT = nStateT
            step += 1
            self.agent.decayingLogStd(step)
            self.oldAgent.decayingLogStd(step)
            if step % (self.updateStep+1) == 0 and self.inferMode == False:
                k += 1
                self.train(step, self.epoch)
                self.clear()
                if k % self.updateOldP == 0:
                    self.oldAgent.update(self.agent)
                    k = 0
            
            if step % 2000 == 0:
                self.LogSucessRate(step)
            
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