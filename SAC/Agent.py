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
                            iSize=self.aData['sSize'][-1]*self.aData['sSize'][0])
                elif netCat == "CNET":
                    self.actor = \
                        CNET(
                            netData, 
                            iSize=self.aData['sSize'][0], 
                            WH=self.aData['sSize'][-1])
                else:
                    RuntimeError("The name of agent is not invalid")
                self.shortcut = netData['shortcut']
                inputDim = self.actor.fSize[self.shortcut-1]
                
            if netName == "critic":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.critic01 = \
                        MLP(
                            netData, 
                            iSize=self.aData['sSize'][-1]*self.aData['sSize'][0]+self.aData['aSize'])
                    self.critic02 = \
                        MLP(
                            netData, 
                            iSize=self.aData['sSize'][-1]*self.aData['sSize'][0]+self.aData['aSize'])
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

            if netName == "policy":
                netData = self.aData[netName]
                netCat = netData['netCat']
                if netCat == "MLP":
                    self.policy = \
                        MLP(
                            netData, 
                            iSize=inputDim)
                else:
                    RuntimeError("The name of agent is not invalid")

        self.temperature = torch.zeros((1), requires_grad=True, device=self.device)
        
    def forward(self, state):

        if torch.is_tensor(state) is False:
            state = torch.tensor(state).float()
        
        state = state.to(self.device)
        state = state.view((state.shape[0], -1))

        aFeature, mean = self.actor.forward(state, shortcut=self.shortcut)
        log_std = self.policy.forward(aFeature)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        cat = torch.cat((state, action), dim=1)
        critic01 = self.critic01(cat)
        critic02 = self.critic02(cat)

        return action, logProb, (critic01, critic02), entropy

    def calLoss(self, state, target, actions,  alpha=0):
        self.actor.train()
        self.critic01.train()
        self.critic02.train()
        self.policy.train()

        state = state.to(self.device)
        state = state.view((state.shape[0], -1)).detach()

        action, logProb, critics, entropy = self.forward(state)
        target1, target2 = target
        stateAction = torch.cat((state, actions), dim=1).detach()

        critic1 = self.critic01(stateAction)
        critic2 = self.critic02(stateAction)

        lossCritic1 = torch.mean((critic1-target1).pow(2))/2
        lossCritic2 = torch.mean((critic2-target2).pow(2))/2

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