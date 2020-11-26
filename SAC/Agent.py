import torch
from baseline.utils import constructNet
from baseline.baseNetwork import baseAgent


class sacAgent(baseAgent):

    def __init__(self, aData):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.keyList = list(self.aData.keys())
        device = self.aData['device']
        self.device = torch.device(device)
        self.getSize()
        self.buildModel()
    
    def getSize(self):
        self.sSize = self.aData['sSize']
        self.aSize = self.aData['aSize']
        self.inputSize1 = self.sSize[0]
        div = 1
        stride = self.aData['CNN1D']['stride']
        fSize = self.aData['CNN1D']['nUnit'][-1]
        for s in stride:
            div *= s
        self.inputSize2 = int((self.sSize[-1]/div)*fSize) + 6
        self.inputSize3 = self.inputSize2 + self.aSize

    def buildModel(self):
        for netName in self.keyList:
            
            if netName == "CNN1D":
                netData = self.aData[netName]
                self.aF = \
                    constructNet(
                        netData,
                        iSize=self.inputSize1
                        )
                self.cF1 = \
                    constructNet(
                        netData,
                        iSize=self.inputSize1
                        )
                self.cF2 = \
                    constructNet(
                        netData,
                        iSize=self.inputSize1
                        )
                        
            if netName == "actor":
                netData = self.aData[netName]
                netCat = netData['netCat']
                self.actor = \
                    constructNet(
                        netData, 
                        iSize=self.inputSize2)

            if netName == "critic":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.critic01 = \
                        constructNet(
                            netData, 
                            iSize=self.inputSize3)
                    self.critic02 = \
                        constructNet(
                            netData, 
                            iSize=self.inputSize3)

        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def forward(self, state):
        
        rState, lidarImg = state

        if torch.is_tensor(rState):
            rState = rState.to(self.device).float()
            lidarImg = lidarImg.to(self.device).float()
        else:
            rState = torch.tensor(rState).to(self.device).float()
            lidarImg = torch.tensor(lidarImg).to(self.device).float()

        if lidarImg.dim() == 2:
            lidarImg = torch.unsqueeze(lidarImg, 0)
            rState = torch.unsqueeze(rState, 0)

        actorFeature = self.aF(lidarImg)
        ss = torch.cat((rState, actorFeature), dim=1)
        output = self.actor(ss)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        cSS1 = self.cF1(lidarImg)
        cSS2 = self.cF2(lidarImg)
        cat1 = torch.cat((action, rState, cSS1), dim=1)
        cat2 = torch.cat((action, rState, cSS2), dim=1)
        critic01 = self.critic01.forward(cat1)
        critic02 = self.critic02.forward(cat2)

        return action, logProb, (critic01, critic02), entropy

    def actorForward(self, state, dMode=False):
        rState, lidarImg = state

        if torch.is_tensor(rState):
            rState = rState.to(self.device).float()
            lidarImg = lidarImg.to(self.device).float()
        else:
            rState = torch.tensor(rState).to(self.device).float()
            lidarImg = torch.tensor(lidarImg).to(self.device).float()

        if lidarImg.dim() == 2:
            lidarImg = torch.unsqueeze(lidarImg, 0)
            rState = torch.unsqueeze(rState, 0)

        actorFeature = self.aF(lidarImg)
        ss = torch.cat((rState, actorFeature), dim=1)
        output = self.actor(ss)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        if dMode:
            action = torch.tanh(mean)
        else:
            gaussianDist = torch.distributions.Normal(mean, std)
            x_t = gaussianDist.rsample()
            action = torch.tanh(x_t) 
        
        return action

    def criticForward(self, state, action):
        rState, lState = state

        cSS1 = self.criticFeature01(lState)
        cSS2 = self.criticFeature02(lState)

        cat1 = torch.cat((action, rState, cSS1), dim=1)
        cat2 = torch.cat((action, rState, cSS2), dim=1)
        critic01 = self.critic01.forward(cat1)
        critic02 = self.critic02.forward(cat2) 
    
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