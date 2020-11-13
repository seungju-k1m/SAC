import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet, getOptim


class sacAgent(baseAgent):

    def __init__(self, aData, oData, device='cpu'):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.optimData = oData
        self.hiddenSize = self.aData['Feature']['hiddenSize']
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()
    
    def to(self, device):
        self.device = device
        self.Feature = self.Feature.to(device)
        self.actor = self.actor.to(device)
        self.critic01 = self.critic01.to(device)
        self.critic02 = self.critic02.to(device)

    def getISize(self):
        self.iFeature01 = self.aData['sSize'][-1]
        self.iFeature02 = self.hiddenSize
        self.iFeature03 = self.hiddenSize + 2
        self.sSize = self.aData['sSize']
    
    def buildModel(self):
        for netName in self.keyList:
            netData = self.aData[netName]
            if netName == 'Feature':
                self.Feature = constructNet(netData, iSize=self.iFeature01)
            elif netName == 'actor':
                self.actor = constructNet(netData, iSize=self.iFeature02)
            elif netName == 'critic':
                self.critic01 = constructNet(netData, iSize=self.iFeature03)
                self.critic02 = constructNet(netData, iSize=self.iFeature03)
        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
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
       
        state = torch.unsqueeze(state, dim=0)
        Feature, (hA, cA) = self.Feature(state, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)

        output = self.actor(Feature)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        FeatureQ = torch.cat((action, Feature), dim=1)

        critic01, critic02 = self.critic01(FeatureQ), self.critic02(FeatureQ)
        return action, logProb, (critic01, critic02), entropy, (hA, cA)
    
    def criticForward(self, state, action, lstmState=None):
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
        state = torch.unsqueeze(state, dim=0)

        Feature, (hA, cA) = self.Feature(state, (hAState, cAState))

        Feature = torch.squeeze(Feature, dim=0)
        FeatureQ = torch.cat((action, Feature), dim=1)

        critic01, critic02 = self.critic01(FeatureQ), self.critic02(FeatureQ)
        return critic01, critic02

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
        state = torch.unsqueeze(state, dim=0)
        Feature, (hA, cA) = self.Feature(state, (hAState, cAState))
        Feature = torch.squeeze(Feature, dim=0)
        output = self.actor(Feature)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)

        return action, (hA, cA)

    def calQLoss(self, state, target, pastActions, lstmState):

        critic1, critic2 = self.criticForward(state, pastActions, lstmState)
        lossCritic1 = torch.mean((critic1-target).pow(2)/2)
        lossCritic2 = torch.mean((critic2-target).pow(2)/2)

        return lossCritic1, lossCritic2
    
    def calALoss(self, state, lstmState, alpha=0):
        action, logProb, critics, entropy, _ = self.forward(state, lstmState)
        critic1, critic2 = critics
        # c1a, c2a = torch.abs(critic1), torch.abs(critic2)

        # ca = torch.cat((c1a, c2a), dim=1)

        # argmin = torch.argmin(ca, dim=1).view(-1, 1)
        # minc = torch.cat((c1a, c2a), dim=1)
        # c = []
        # for z, i in enumerate(argmin):
        #     c.append(minc[z, i])
        # critic = torch.stack(c, dim=0)
        critic = torch.min(critic1, critic2)
    
        if alpha != 0:
            tempDetached = alpha
        else:
            tempDetached = self.temperature.exp().detach()
        lossPolicy = torch.mean(tempDetached * logProb - critic)
        detachedLogProb = logProb.detach()
        lossTemp = \
            torch.mean(
                self.temperature.exp()*(-detachedLogProb+self.aData['aSize'])
            )

        return lossPolicy, lossTemp

    def calLoss(self, state, target, pastActions,  alpha=0):
        state = state.to(self.device)
        state = state.view((state.shape[0], -1)).detach()

        # 1. Calculate the loss of the Critics.
        # state and actions are derived from the replay memory.
        stateAction = torch.cat((state, pastActions), dim=1).detach()
        critic1 = self.critic01(stateAction)
        critic2 = self.critic02(stateAction)

        lossCritic1 = torch.mean((critic1-target).pow(2)/2)
        lossCritic2 = torch.mean((critic2-target).pow(2)/2)

        # 2. Calculate the loss of the Actor.
        state = state.detach()
        action, logProb, critics, entropy = self.forward(state)
        critic1_p, critic2_p = critics
        critic_p = torch.min(critic1_p, critic2_p)

        if alpha != 0:
            tempDetached = alpha
        else:
            self.temperature.train()
            tempDetached = self.temperature.exp().detach()
        detachedLogProb = logProb.detach()
        lossPolicy = torch.mean(tempDetached * logProb - critic_p)
        lossTemp = \
            torch.mean(
                self.temperature.exp()*(-detachedLogProb+self.aData['aSize']))
        
        return lossCritic1, lossCritic2, lossPolicy, lossTemp