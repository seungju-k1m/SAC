import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet, getOptim


class sacAgent(baseAgent):

    def __init__(self, aData, oData, device='cpu'):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.optimData = oData
        self.hiddenSize = self.aData['actorFeature02']['hiddenSize']
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()
    
    def to(self, device):
        self.device = device
        self.actorFeature01 = self.actorFeature01.to(device)
        self.actorFeature02 = self.actorFeature02.to(device)
        self.actor = self.actor.to(device)

        self.criticFeature01_1 = self.criticFeature01_1.to(device)
        self.criticFeature02_1 = self.criticFeature02_1.to(device)
        self.critic01 = self.critic01.to(device)

        self.criticFeature01_2 = self.criticFeature01_2.to(device)
        self.criticFeature02_2 = self.criticFeature02_2.to(device)
        self.critic02 = self.critic02.to(device)

    def getISize(self):
        self.iFeature01 = self.aData['sSize'][-1]
        temp = self.aData['actorFeature01']['stride']
        div = 1
        for t in temp:
            div *= t
        nUnit = self.aData['actorFeature01']['nUnit'][-1]
        self.iFeature02 = int((self.iFeature01/div)**2) * nUnit
        self.iFeature03 = self.hiddenSize
        self.sSize = self.aData['sSize']
    def criticStep(self, globalAgent):
        cF1_1, cF2_1, c1, cF1_2, cF2_2, c2 = (
            globalAgent.criticFeature01_1,
            globalAgent.criticFeature02_1,
            globalAgent.critic01,

            globalAgent.criticFeature01_2,
            globalAgent.criticFeature02_2,
            globalAgent.critic02,
        )
        
        cF1_1.grad = self.criticFeature01_1.grad
    
    def actorStep(self):
        self.aOptim.step()
        self.aFOptim01.step()
        self.aFOptim02.step()

    def calculateNorm(self):
        pass
    
    def buildModel(self):
        for netName in self.keyList:
            netData = self.aData[netName]
            if netName == 'actorFeature01':
                self.actorFeature01 = constructNet(netData, iSize=self.sSize[0], WH=self.iFeature01)
            elif netName == 'actorFeature02':
                self.actorFeature02 = constructNet(netData, iSize=self.iFeature02)
            elif netName == 'actor':
                self.actor = constructNet(netData, iSize=self.iFeature03)
            elif netName == 'criticFeature01':
                self.criticFeature01_1 = constructNet(
                    netData, iSize=self.sSize[0], WH=self.iFeature01)
                self.criticFeature01_2 = constructNet(
                    netData, iSize=self.sSize[0], WH=self.iFeature01)
            elif netName == 'criticFeature02':
                self.criticFeature02_1 = constructNet(netData, iSize=self.iFeature02)
                self.criticFeature02_2 = constructNet(netData, iSize=self.iFeature02)
            elif netName == 'critic':
                self.critic01 = constructNet(netData, iSize=self.iFeature03)
                self.critic02 = constructNet(netData, iSize=self.iFeature03)
        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def forward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape : [batch, 1, 96, 96]
            lstmState:[tuple]
                dtype : ((hAState, cAState), (hCSTate01, cCState01),(hCSTate02, cCState02)), each state is tensor
                shape : [1, batch, 512] for each state
                default : None
                if default is None, the lstm State is padded.
            stateAction:[tensor]
                shape : [batch, 1, 96, 96]
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
        if lstmState is None:
            hAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cAState = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hAState, cAState = lstmState[0]
            hCState01, cCState01 = lstmState[1]
            hCState02, cCState02 = lstmState[2]
            hAState, cAState = hAState.to(self.device), cAState.to(self.device)
            hCState01, cCState01 = hCState01.to(self.device), cCState01.to(self.device)
            hCState02, cCState02 = hCState02.to(self.device), cCState02.to(self.device)
       
        aF1 = self.actorFeature01(state)
        aF1 = torch.unsqueeze(aF1, dim=1)
        aL1, (hA, cA) = self.actorFeature02(aF1, (hAState, cAState))
        output = self.actor(aL1)
        output = torch.squeeze(output, dim=1)

        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        stateAction = state
        stateAction[:, :, :, 6:8] = action

        cSS1_1 = torch.unsqueeze(self.criticFeature01_1(stateAction), dim=1)
        cSS1_2 = torch.unsqueeze(self.criticFeature01_2(stateAction), dim=1)

        cSS2_1, (hC1, cC1) = self.criticFeature02_1(cSS1_1, (hCState01, cCState01))
        cSS2_2, (hC2, cC2) = self.criticFeature02_1(cSS1_2, (hCState02, cCState02))

        critic01, critic02 = self.critic01(cSS2_1), self.critic02(cSS2_2)

        return action, logProb, (critic01, critic02), entropy, ((hA, cA), (hC1, cC1), (hC2, cC2))
    
    def criticForward(self, state, action, lstmState=None):
        """
        input:
            state:[tensor]
                shape:[batch, 1, 96, 96]
            action:[tensor]
                shape:[batch, 2]
            lstmState:[tuple]
                dtype:((hs1, cs1), (hs2, cs2))
                shape :[1, bath, 512] for each state
        output:
            critics:[tuple]
                dtype:(critic01, critic02)
                shape:[batch, 1] for each state
        """
        state = state.to(self.device)
        bSize = state.shape[0]
        if lstmState is None:
            hCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState01 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            hCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
            cCState02 = torch.zeros(1, bSize, self.hiddenSize).to(self.device)
        else:
            hCState01, cCState01 = lstmState[1]
            hCState02, cCState02 = lstmState[2]
            hCState01, cCState01 = hCState01.to(self.device), cCState01.to(self.device)
            hCState02, cCState02 = hCState02.to(self.device), cCState02.to(self.device)
       
        state[:, :, :, 6:8] = action

        cF1_1 = self.criticFeature01_1.forward(state)
        cF2_1 = self.criticFeature02_1.forward(cF1_1, (hCState01, cCState01))
        c1 = self.critic01.forward(cF2_1)

        cF1_2 = self.criticFeature01_2.forward(state)
        cF2_2 = self.criticFeature02_2.forward(cF1_2, (hCState02, cCState02))
        c2 = self.critic01.forward(cF2_2)
 
        return c1, c2

    def actorForward(self, state, lstmState=None):
        """
        input:
            state:[tensor]
                shape:[batch, 1, 96, 96]
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
     
        aF1 = self.actorFeature01(state)
        aL1, (hA, cA) = self.actorFeature02(aF1, (hAState, cAState))
        output = self.actor(aL1)

        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)

        return action, (hA, cA)

    def calQLoss(self, state, target, pastActions):

        critic1, critic2 = self.criticForward(state, pastActions)
        lossCritic1 = torch.mean((critic1-target).pow(2)/2)
        lossCritic2 = torch.mean((critic2-target).pow(2)/2)

        return lossCritic1, lossCritic2
    
    def calALoss(self, state, alpha=0):

        action, logProb, critics, entropy, _ = self.forward(state)
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