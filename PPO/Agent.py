import math
import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet


class ppoAgent(baseAgent):

    def __init__(
        self, 
        aData,
        oData,
        coeff=0.01,
        epsilon=0.2,
        logStd=-1.6,
        fixedSigma=False,
        device='cpu'
        ):
        super(ppoAgent, self).__init__()
        
        self.fixedSigma = fixedSigma
        self.aData = aData
        self.optimData = oData
        self.coeff = coeff
        self.epsilon = epsilon
        self.logStd = logStd
        self.hiddenSize = self.aData['LSTM']['hiddenSize']
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()

    def update(self, Agent):
        self.CNN.load_state_dict(Agent.CNN.state_dict())
        self.actor.load_state_dict(Agent.actor.state_dict())
        self.LSTM.load_state_dict(Agent.LSTM.state_dict())
        self.critic.load_state_dict(Agent.critic.state_dict())
    
    def to(self, device):
        self.device = device
        self.LSTM = self.LSTM.to(device)
        self.CNN = self.CNN.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def getISize(self):
        self.iFeature01 = self.aData['sSize'][-1]
        self.iFeature02 = self.hiddenSize
        self.sSize = self.aData['sSize']
        div = 1
        stride = self.aData['CNN']['stride']
        fSize = self.aData['CNN']['nUnit'][-1]
        for s in stride:
            div *= s
        self.iFeature03 = int((self.sSize[-1]/div)**2*fSize)
    
    def buildModel(self):
        for netName in self.keyList:
            netData = self.aData[netName]
            if netName == 'CNN':
                self.CNN = constructNet(netData, iSize=1, WH=96)
            elif netName == 'LSTM':
                self.LSTM = constructNet(netData, iSize=self.iFeature03)
            elif netName == 'actor':
                self.actor = constructNet(netData, iSize=self.iFeature02)
            elif netName == 'critic':
                self.critic = constructNet(netData, iSize=self.iFeature02)
    
    def forward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape : [batch, 726]
            lstmState:[tuple]
                dtype : (hAState, cAState), each state is tensor
                shape : [1, batch, 512] for each state
                default : None
                if default is None, the lstm State is padded.
        output:
            action:[tensor]
                shape : [batch, actionSize]
            logProb:[tensor]
                shape : [batch, 1]
            critics:[tuple]
                dtype : (critic01, critic02)
                shape : [batch, 1] for each state
            entropy:[tensor]
                shape : [batch, 1]
            lstmState:[tuple]
                dtype : (hA, cA)
                shape : [1, batch, hiddenSize] for each state
        """
        bSize = state.shape[0]
        state = state.to(self.device)
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hAState, cAState = lstmState
        
        cnnF = self.CNN(state)
        cnnF = torch.unsqueeze(cnnF, dim=0)
       
        Feature, (hA, cA) = self.LSTM(cnnF, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)

        if self.fixedSigma:
            mean = self.actor(Feature)
            std = math.exp(self.logStd)
        else:
            output = self.actor(Feature)
            mean, logStd = output[:, :2], output[:, 2:]
            std = torch.exp(logStd)

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        critic = self.critic(Feature)
        return action, logProb, critic, entropy, (hA, cA)
    
    def criticForward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape:[batch, 726]
            action:[tensor]
                shape:[batch, 2]
            lstmState:[tuple]
                shape:[1, batch, 512]
        output:
            critics:[tuple]
                dtype:(critic01, critic02)
                shape:[batch, 1] for each state
        """
        bSize = state.shape[0]
        state = state.to(self.device)
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hAState, cAState = lstmState
        cnnF = self.CNN(state)
        cnnF = torch.unsqueeze(cnnF, dim=0)
       
        Feature, (hA, cA) = self.LSTM(cnnF, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)

        critic = self.critic(Feature)
        return critic

    def actorForward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape:[batch, 726]
            lstmState:[tuple]
                dtype:(hs1, cs1)
                shape :[1, bath, 512] for each state
        output:
            action:[tensor]
                shape:[batch, 2]
            lstmState:[tuple]
                dtype:(hs1, cs1)
                shape :[1, bath, 512] for each state
        """
        state = state.to(self.device)
        bSize = state.shape[0]
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hAState, cAState = lstmState
            hAState, cAState = hAState.to(self.device), cAState.to(self.device)
        cnnF = self.CNN(state)
        cnnF = torch.unsqueeze(cnnF, dim=0)
       
        Feature, (hA, cA) = self.LSTM(cnnF, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)
        if self.fixedSigma:
            mean = self.actor(Feature)
            std = math.exp(self.logStd)
        else:
            output = self.actor(Feature)
            mean, logStd = output[:, :2], output[:, 2:]
            std = torch.exp(logStd)
        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)

        return action, (hA, cA)

    def calQLoss(self, state, lstmState, target):

        critic = self.criticForward(state, lstmState)
        lossCritic1 = torch.mean((critic-target).pow(2)/2)

        return lossCritic1
    
    def calAObj(self, old_agent, state, lstmState, action, gae):
        
        prob, entropy = self.calLogProb(state, action, lstmState=lstmState)
        oldProb, _ = old_agent.calLogProb(state, action, lstmState=lstmState)
        oldProb = oldProb.detach()
        ratio = prob / oldProb
        obj = torch.min(ratio * gae, torch.clamp(ratio, 1-self.epsilon, 1.2) * gae) + self.coeff * entropy

        return (-obj).mean(), entropy

    def calLogProb(self, state, action, lstmState=None):
        state = state.to(self.device)
        bSize = state.shape[0]
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hAState, cAState = lstmState
            hAState, cAState = hAState.to(self.device), cAState.to(self.device)
        cnnF = self.CNN(state)
        cnnF = torch.unsqueeze(cnnF, dim=0)
       
        Feature, (hA, cA) = self.LSTM(cnnF, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)
        if self.fixedSigma:
            mean = self.actor(Feature)
            std = math.exp(self.logStd)
        else:
            output = self.actor(Feature)
            mean, logStd = output[:, :2], output[:, 2:]
            std = torch.exp(logStd)
        gaussianDist = torch.distributions.Normal(mean, std)
        x = torch.atanh(action)
        log_prob = gaussianDist.log_prob(x).sum(1, keepdim=True)
        log_prob -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = gaussianDist.entropy().sum(1, keepdim=True)
        # entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        return log_prob.exp(), entropy.mean()

