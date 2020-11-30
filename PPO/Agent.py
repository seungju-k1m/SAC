import torch
from baseline.utils import constructNet
from baseline.baseNetwork import baseAgent, LSTMNET


class ppoAgent(baseAgent):

    def __init__(
        self, 
        aData,
        coeff=0.01,
        epsilon=0.2,
        device='cpu'
        ):
        super(ppoAgent, self).__init__()
        
        self.aData = aData
        self.keyList = list(self.aData.keys())
        self.device = torch.device(device)
        self.buildModel()
        self.to(self.device)
        self.coeff = coeff
        self.epsilon = epsilon
    
    def buildModel(self):
        for netName in self.keyList:
            if netName == "actor":
                netData = self.aData[netName]
                self.actor = AgentV1(netData)

            if netName == "critic":
                netData = self.aData[netName]
                self.critic = AgentV1(netData)
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def forward(self, state):
        output = self.actor.forward(state)[0]
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)
        critic = self.criticForward(state, action)

        return action, logProb, entropy, critic

    def actorForward(self, state, dMode=False):

        output = self.actor.forward(state)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        if dMode:
            action = torch.tanh(mean)
        else:
            gaussianDist = torch.distributions.Normal(mean, std)
            x_t = gaussianDist.rsample()
            action = torch.tanh(x_t)
        
        return action

    def criticForward(self, state, action):
        state = list(state)
        state.append(action)
        state = tuple(state)
        
        critic = self.critic.forward(state)[0]
    
        return critic

    def calQLoss(self, state, target, pastActions):

        critic = self.criticForward(state, pastActions)
        lossCritic = torch.mean((critic-target).pow(2)/2)

        return lossCritic
    
    def calAObj(self, old_agent, state, action, gae):
        prob, entropy = self.calLogProb(state, action)
        oldProb, _ = old_agent.calLogProb(state, action)
        oldProb = oldProb.detach()
        ratio = prob / (oldProb + 1e-4)
        obj = torch.min(ratio * gae, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * gae) + self.coeff * entropy

        return (-obj).mean(), entropy

    def calLogProb(self, state, action):
        
        output = self.actor.forward(state)
        mean, log_std = output[:, :self.aData["aSize"]], output[:, self.aData["aSize"]:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        gaussianDist = torch.distributions.Normal(mean, std)
        x = torch.atanh(action)
        log_prob = gaussianDist.log_prob(x).sum(1, keepdim=True)
        log_prob -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = gaussianDist.entropy().sum(1, keepdim=True)
        return log_prob.exp(), entropy.mean()

    def update(self, Agent):
        self.actor.updateParameter(Agent.actor, tau=1.0)
        self.critic.updateParameter(Agent.critic, tau=1.0)


class AgentV1:

    def __init__(
        self,
        mData
    ):
        self.mData = mData
        self.moduleNames = list(self.mData.keys())
        self.moduleNames.sort()
        self.buildModel()
    
    def buildModel(self):
        self.model = {}
        self.connect = {}
        for _ in range(10):
            self.connect[_] = []
        
        for name in self.moduleNames:
            self.model[name] = constructNet(self.mData[name])
            if 'input' in self.mData[name].keys():
                inputs = self.mData[name]['input']
                for i in inputs:
                    self.connect[i].append(name)
    
    def buildOptim(self):
        listLayer = []
        for name in self.moduleNames:
            layer = self.model[name]
            if type(layer) is not None:
                listLayer.append(layer)
        return tuple(listLayer)
    
    def updateParameter(self, Agent, tau):

        with torch.no_grad():
            for name in self.moduleNames:
                parameters = self.model[name].parameters()
                tParameters = Agent.model[name].parameters()
                for p, tp in zip(parameters, tParameters):
                    p.copy_((1 - tau) * p + tau * tp)
    
    def calculateNorm(self):
        totalNorm = 0
        for name in self.moduleNames:
            parameters = self.model[name].parameters()
            for p in parameters:
                norm = p.grad.data.norm(2)
                totalNorm += norm
        
        return totalNorm

    def to(self, device):
        for name in self.moduleNames:
            self.model[name].to(device)

    def forward(self, inputs):
        inputSize = len(inputs)
        stopIdx = []

        for i in range(inputSize):
            idx = self.connect[i]
            for i in idx:
                stopIdx.append(self.moduleNames.index(i))
        
        flow = 0
        forward = None
        output = []
        while 1:
            name = self.moduleNames[flow]
            layer = self.model[name]
            if flow in stopIdx:
                nInputs = self.mData[name]['input']
                for nInput in nInputs:
                    if forward is None:
                        forward = inputs[nInput]
                    else:
                        if type(forward) == tuple:
                            forward = list(forward)
                            forward.append(inputs[nInput])
                            forward = tuple(forward)
                        else:
                            forward = (forward, inputs[nInput])
            
            if layer is None:
                pass
            else:
                if type(layer) == LSTMNET:
                    forward, lstmState = layer.forward(forward)
                    output.append(lstmState)
                else:
                    forward = layer.forward(forward)

            if "output" in self.mData[name].keys():
                if self.mData[name]['output']:
                    output.append(forward)
            flow += 1
            if flow == len(self.moduleNames):
                break
        return tuple(output)