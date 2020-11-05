import torch
from baseline.baseNetwork import MLP, CNET, baseAgent


class sacAgent(baseAgent):

    def __init__(self, aData):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.keyList = list(self.aData.keys())
        device = self.aData['device']
        self.device = torch.device(device)
        self.buildModel()

    def buildModel(self):
        for netName in self.keyList:
            
            if netName == "actor":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.actor = \
                        MLP(
                            netData, 
                            iSize=int((self.aData['sSize'][-1]/8)**2 * self.aData['actorFeature']['nUnit'][-1]) + 6)
                        
                elif netCat == "CNET":
                    self.actor = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                else:
                    RuntimeError("The name of agent is not invalid")
            
            if netName == "actorFeature":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.actorFeature = \
                        MLP(
                            netData, 
                            iSize=self.aData['sSize'][-1]/16 + 6)
                elif netCat == "CNET":
                    self.actorFeature = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                else:
                    RuntimeError("The name of agent is not invalid")
                
            if netName == "critic":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.critic01 = \
                        MLP(
                            netData, 
                            iSize=int((self.aData['sSize'][-1]/8)**2 * self.aData['actorFeature']['nUnit'][-1]) + 6 + self.aData['aSize'])
                    self.critic02 = \
                        MLP(
                            netData, 
                            iSize=int((self.aData['sSize'][-1]/8)**2 * self.aData['actorFeature']['nUnit'][-1]) + 6 + self.aData['aSize'])
                elif netCat == "CNET":
                    self.critic01 = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                    self.critic02 = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                else:
                    RuntimeError("The name of agent is not invalid")
            if netName == "criticFeature":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.criticFeature01 = \
                        MLP(
                            netData, 
                            iSize=int(self.aData['sSize'][-1]/8 * netData['fSize'][-1]) + 6 + self.aData['aSize'])
                    self.criticFeature02 = \
                        MLP(
                            netData, 
                            iSize=int(self.aData['sSize'][-1]/8 * netData['fSize'][-1]) + 6 + self.aData['aSize'])
                elif netCat == "CNET":
                    self.criticFeature01 = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                    self.criticFeature02 = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                else:
                    RuntimeError("The name of agent is not invalid")

        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def forward(self, state):
        
        state, lidarImg = state
        state = state.to(self.device).float()
        lidarImg = lidarImg.to(self.device).float()

        if lidarImg.dim() == 3:
            lidarImg = torch.unsqueeze(lidarImg, 0)
            state = torch.unsqueeze(state, 0)

        actorFeature = self.actorFeature(lidarImg)
        ss = torch.cat((state, actorFeature), dim=1)
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

        cSS1 = self.criticFeature01(lidarImg)
        cSS2 = self.criticFeature02(lidarImg)
        cat1 = torch.cat((action, state, cSS1), dim=1)
        cat2 = torch.cat((action, state, cSS2), dim=1)
        critic01 = self.critic01.forward(cat1)
        critic02 = self.critic02.forward(cat2)

        return action, logProb, (critic01, critic02), entropy
    
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