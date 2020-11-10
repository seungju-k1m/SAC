import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet


class sacAgent(baseAgent):

    def __init__(self, aData, oData, device='cpu'):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.optimData = oData
        self.hiddenSize = self.aData['actorFeature01']['hiddenSize']
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()
    
    def to(self, device):
        self.device = device
        self.actorFeature01 = self.actorFeature01.to(device)
        self.actor = self.actor.to(device)

        self.criticFeature01 = self.criticFeature01.to(device)
        self.critic01 = self.critic01.to(device)

        self.criticFeature02 = self.criticFeature02.to(device)
        self.critic02 = self.critic02.to(device)

    def getISize(self):
        self.iFeature01 = self.aData['sSize'][-1]
        self.iFeature02 = self.hiddenSize
        self.sSize = self.aData['sSize']
    
    def buildModel(self):
        
        for netName in self.keyList:
            netData = self.aData[netName]
            if netName == 'actorFeature01':
                self.actorFeature01 = constructNet(netData, iSize=self.iFeature01)
            
            if netName == 'actor':
                self.actor = constructNet(netData, iSize=self.iFeature02)
            if netName == 'criticFeature01':
                self.criticFeature01 = constructNet(netData, iSize=self.iFeature01+2)
                self.criticFeature02 = constructNet(netData, iSize=self.iFeature01+2)
            if netName == 'critic':
                self.critic01 = constructNet(netData, iSize=self.iFeature02)
                self.critic02 = constructNet(netData, iSize=self.iFeature02)
        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def forward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape : [batch, 726]
            lstmState:[tuple]
                dtype : ((hAState, cAState), (hCState01, cCState01), (hCState02, cCState02))
                shape : [1, batch, 512] for each state
                default : None
                if default is None, the lstm State is zero padded.
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
                dtype : ((hA, cA), (hC1, cC1), (hC2, cC2))
                shape : [1, batch, hiddenSize] for each state
        """
        bSize = state.shape[0]
        state = state.to(self.device)
        state = torch.unsqueeze(state, dim=0)
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            (hAState, cAState), (hCState01, cCState01), (hCState02, cCState02) = lstmState
       
        aF1, (hA, cA) = self.actorFeature01(state, (hAState, cAState))
        aF1 = torch.squeeze(aF1, dim=0)
        output = self.actor(aF1)
        
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)
        action = torch.unsqueeze(action, dim=0)
        stateAction = torch.cat((action, state), dim=-1)

        cSS1, (hC1, cC1) = self.criticFeature01(stateAction, (hCState01, cCState01))
        cSS2, (hC2, cC2) = self.criticFeature02(stateAction, (hCState02, cCState02))

        cSS1 = torch.squeeze(cSS1, dim=0)
        cSS2 = torch.squeeze(cSS2, dim=0)

        action = torch.squeeze(action, dim=0)

        critic01, critic02 = self.critic01(cSS1), self.critic02(cSS2)
        return action, logProb, (critic01, critic02), entropy, ((hA, cA), (hC1, cC1), (hC2, cC2))
    
    def criticForward(self, state, action, lstmState=None):
        """
        input:
            state:[tensor]
                shape:[batch, 726]
            action:[tensor]
                shape:[batch, 2]
            lstmState:[tuple]
                shape:[batch, hiddenSize]
        output:
            critics:[tuple]
                dtype:(critic01, critic02)
                shape:[batch, 1] for each state
        """
        bSize = state.shape[0]
        state = state.to(self.device)
        stateAction = torch.cat((action, state), dim=1)
        stateAction = torch.unsqueeze(stateAction, dim=0)
        if lstmState is None:
            hCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            (hCState01, cCState01), (hCState02, cCState02) = lstmState

        cSS1, (hC1, cC1) = self.criticFeature01(stateAction, (hCState01, cCState01))
        cSS2, (hC2, cC2) = self.criticFeature02(stateAction, (hCState02, cCState02))

        critic01, critic02 = torch.squeeze(self.critic01(cSS1), dim=0), torch.squeeze(self.critic02(cSS2), dim=0)
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
     
        aF1, (hA, cA) = self.actorFeature01(state, (hAState, cAState))
        aF1 = torch.squeeze(aF1, dim=0)
        output = self.actor(aF1)

        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)

        return action, (hA, cA)

    def calQLoss(self, state, target, pastActions, lstmState):

        critic1, critic2 = self.criticForward(state, pastActions, lstmState=lstmState)
        lossCritic1 = torch.mean((critic1-target).pow(2)/2)
        lossCritic2 = torch.mean((critic2-target).pow(2)/2)

        return lossCritic1, lossCritic2
    
    def calALoss(self, state, lstmState, alpha=0):
        action, logProb, critics, entropy, _ = self.forward(state, lstmState)
        critic1, critic2 = critics
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