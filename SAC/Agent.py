import torch
from baseline.baseNetwork import baseAgent
from baseline.utils import constructNet, getOptim


class sacAgent(baseAgent):

    def __init__(self, aData, oData, device='cpu'):
        super(sacAgent, self).__init__()
        
        self.aData = aData
        self.optimData = oData
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.getISize()
        self.buildModel()
    
    def to(self, device):
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
        self.iFeature01 = 80
        self.iFeature02 = 100
        self.iFeature03 = 30
        self.offset = 10
    
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
        
        if type(state) == list:
            lidarImg, rState, hState, cState = [], [], [], []
            for s in state:
                lidarImg.append(s[0])
                rState.append(s[1])
                hState.append(s[2])
                cState.append(s[3])
            lidarImg = torch.stack(lidarImg, dim=0).to(self.device).float()
            rState = torch.stack(rState, dim=0).to(self.device).float()
            hState = torch.stack(hState, dim=0).to(self.device).float()
            cState = torch.stack(cState, dim=0).to(self.device).float() 
        else:
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
        if type(state) == list:
            lidarImg, rState, hState, cState = [], [], [], []
            for s in state:
                lidarImg.append(s[0])
                rState.append(s[1])
                hState.append(s[2])
                cState.append(s[3])
            lidarImg = torch.stack(lidarImg, dim=0).to(self.device).float()
            rState = torch.stack(rState, dim=0).to(self.device).float()
            hState = torch.stack(hState, dim=0).to(self.device).float()
            cState = torch.stack(cState, dim=0).to(self.device).float() 
        else:
            lidarImg, rState, (hState, cState) = state

            if torch.is_tensor(lidarImg) is False:
                lidarImg = torch.tensor(lidarImg).to(self.device)
                rState = torch.tensor(rState).to(self.device)
                hState, cState =\
                    torch.tensor(hState).to(self.device), torch.tensor(cState).to(self.device)

        cF1_1 = self.criticFeature01_1.forward(lidarImg)
        cat1 = torch.cat((rState, cF1_1), dim=1)
        cF2_1 = self.criticFeature02_1.forward(cat1)
        c1 = self.critic01.forward(cF2_1)

        cF1_2 = self.criticFeature01_2.forward(lidarImg)
        cat2 = torch.cat((rState, cF1_2), dim=1)
        cF2_2 = self.criticFeature02_2.forward(cat2)
        c2 = self.critic01.forward(cF2_2)
 
        return c1, c2

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