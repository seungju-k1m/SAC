import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet, getOptim


class sacAgent(baseAgent):

    def __init__(self, aData, oData):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.optimData = oData
        self.keyList = list(self.aData.keys())
        device = self.aData['device']
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()
    
    def getISize(self):
        self.iFeature01 = 80
        self.iFeature02 = 100
        self.iFeature03 = 30
        self.offset = 10
    
    def criticStep(self):
        pass
    
    def actorStep(self):
        pass

    def calculateNorm(self):
        pass
    
    def genOptim(self):
        optimKeyList = list(self.optimData.keys())
        for optimKey in optimKeyList:
            if optimKey == 'actor':
                self.aOptim = getOptim(self.optimData[optimKey], self.actor)
                self.aFOptim01 = getOptim(self.optimData[optimKey], self.actorFeature01)
                self.aFOptim02 = getOptim(self.optimData[optimKey], self.actorFeature02)
            if optimKey == 'critic':
                self.cOptim1 = getOptim(self.optimData[optimKey], self.critic01)
                self.cFOptim1_1 = getOptim(self.optimData[optimKey], self.criticFeature01_1)
                self.cFOptim2_1 = getOptim(self.optimData[optimKey], self.criticFeature02_1)
                self.cOptim2 = getOptim(self.optimData[optimKey], self.critic02)
                self.cFOptim1_2 = getOptim(self.optimData[optimKey], self.criticFeature01_2)
                self.cFOptim2_2 = getOptim(self.optimData[optimKey], self.criticFeature02_2)
            if optimKey == 'temperature':
                if self.fixedTemp is False:
                    self.tOptim = getOptim(self.optimData[optimKey], [self.tempValue], floatV=True)

    def buildModel(self):
        for netName in self.keyList:
            netData = self.aData[netName]
            if netName == 'actorFeature01':
                self.actorFeature01 = constructNet(netData, iSize=self.iFeature01)
            elif netName == 'actorFeature02':
                self.actorFeature02 = constructNet(netData, iSize=self.iFeature02)
            elif netName == 'actor':
                self.actor = constructNet(netData, iSize=self.iFeature03)
            elif netName == 'criticFeature01':
                self.criticFeature01_1 = constructNet(netData, iSize=self.iFeature01)
                self.criticFeature01_2 = constructNet(netData, iSize=self.iFeature01)
            elif netName == 'ciriticFeature02':
                self.criticFeature02_1 = constructNet(netData, iSize=self.iFeature02 + self.offset)
                self.criticFeature02_1 = constructNet(netData, iSize=self.iFeature02 + self.offset)
            elif netName == 'critic':
                self.critic01 = constructNet(netData, iSize=self.iFeature03)
                self.critic02 = constructNet(netData, iSize=self.iFeature03)
        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def forward(self, state):
        
        lidarImg, rState, (hState, cState) = state

        if torch.is_tensor(lidarImg) is False:
            lidarImg = torch.tensor(lidarImg).to(self.device)
            rState = torch.tensor(rState).to(self.device)
            hState, cState =\
                torch.tensor(hState).to(self.device), torch.tensor(cState).to(self.device)

        aF1 = self.actorFeature01(lidarImg)
        aF1 = torch.cat((rState, aF1), dim=1)
        aL1, (h0, c0) = self.actorFeature02(aF1)
        output = self.actor(aL1)

        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        cSS1 = self.criticFeature01(lidarImg)
        cSS2 = self.criticFeature02(lidarImg)
        cat1 = torch.cat((action, state, cSS1), dim=1)
        cat2 = torch.cat((action, state, cSS2), dim=1)
        critic01 = self.critic01.forward(cat1)
        critic02 = self.critic02.forward(cat2)

        return action, logProb, (critic01, critic02), entropy
    
    def criticForward(self, state, action):
        lidarImg, rState, (hState, cState) = state

        if torch.is_tensor(lidarImg) is False:
            lidarImg = torch.tensor(lidarImg).to(self.device)
            rState = torch.tensor(rState).to(self.device)
            hState, cState =\
                torch.tensor(hState).to(self.device), torch.tensor(cState).to(self.device)
 
        return critic01, critic02

    def calQLoss(self, state, target, pastActions):

        critic1, critic2 = self.criticForward(state, pastActions)
        lossCritic1 = torch.mean((critic1-target).pow(2)/2)
        lossCritic2 = torch.mean((critic2-target).pow(2)/2)

        return lossCritic1, lossCritic2
    
    def calALoss(self, state, alpha=0):

        action, logProb, critics, entropy = self.forward(state)
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
        self.actor.train()
        self.critic01.train()
        self.critic02.train()

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