import torch
import torch.nn as nn
from baseline.utils import constructNet
from baseline.baseNetwork import baseAgent


class ppoAgent(baseAgent):
    """
    ppoAgent는 ppo algorithm을 지원하는 agent를 생성하기 위한 class이다.

    ppoAgent는 actor, critic을 포함하고 있으며, forwarding methods를 통해 output을 반환할 수 있다.

    ppo에서 요구되는 objective function을 계산하고 이를 반환하며 후에 trainer에서 update가 이루어진다.

    또한, LSTM을 지원하기 위해 cell state control method를 제공한다.

    def __init__:

        args:
            aData:[dict],model을 build하기 위한 configuration
            coeff:[float], entropy regularzation
            epsilon:[float], hyper-parameter for PPO
            device:[float], device 선언/ "cpu", "cuda:0"..
            initLogStd:[float], 분산을 decaying할때, 초기 log[std]
            finLogStd:[float], 최종 log[std]
            annealingStep:[float], init에서 fin까지 도달할때 동안 step의 수
            LSTMNum:[float],
                
                LSTMNum은 LSTM이 몇 번째 module인지 명시한다.
                이는 cell state control을 위해 필요하다.
    """

    def __init__(
        self, 
        aData,
        coeff=0.01,
        epsilon=0.2,
        device='cpu',
        initLogStd=0,
        finLogStd=-1,
        annealingStep=1e6,
        LSTMName=-1
    ):
        super(ppoAgent, self).__init__()
        
        # configuration for PPO Agent
        self.logStd = initLogStd
        self.finLogTsd = finLogStd
        self.deltaStd = (self.logStd - self.finLogTsd)/annealingStep
        self.annealingStep = annealingStep
        self.aData = aData
        self.LSTMName = LSTMName
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
                self.actor = AgentV2(netData, LSTMName=self.LSTMName)

            if netName == "critic":
                netData = self.aData[netName]
                self.critic = AgentV2(netData, LSTMName=self.LSTMName)
    
    def to(self, device):
        """
        actor와 critic에 장치(cpu, cuda:0)를 부착한다.
        """
        self.actor.to(device)
        self.critic.to(device)

    def forward(self, state):
        """
        state를 입력을 받아, action, logprob entropy, crtici을 output으로 반환한다.

        args:
            state:[tuple]
        """
        # 평균과 표준편차를 구한다.
        mean = self.actor.forward(state)[0]
        std = self.logStd.exp()

        # 이를 기반으로 gaussian sampling을 통해 action을 구한다.
        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)

        # log prob를 구한다.
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

        # criticForward를 통해 critic을 구한다.
        critic = self.criticForward(state)

        return action, logProb, entropy, critic

    def actorForward(self, state, dMode=False):

        mean = self.actor.forward(state)[0]
        std = self.logStd.exp()
        if dMode:
            action = torch.tanh(mean)
        else:
            gaussianDist = torch.distributions.Normal(mean, std)
            x_t = gaussianDist.rsample()
            action = torch.tanh(x_t)
        
        return action

    def criticForward(self, state):
        critic = self.critic.forward(state)[0]
        return critic

    def calQLoss(self, state, target):
        """
        critic을 위한 objective function을 구한다.
        """
        critic = self.criticForward(state)
        lossCritic = torch.mean((critic-target).pow(2)/2)

        return lossCritic
    
    def calAObj(self, old_agent, state, action, gae):
        """
        actor를 위한 objective function을 구한다.
        """
        prob, entropy = self.calLogProb(state, action)
        oldProb, _ = old_agent.calLogProb(state, action)
        oldProb = oldProb.detach()
        ratio = prob / (oldProb + 1e-4)
        obj = torch.min(ratio * gae, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * gae) + self.coeff * entropy

        return (-obj).mean(), entropy

    def calLogProb(self, state, action):
        """
        log pi(action|state)를 반환한다.
        """

        mean = self.actor.forward(state)[0]
        # action = torch.clamp(action, -0.9999, 0.9999)
        std = self.logStd.exp()
        gaussianDist = torch.distributions.Normal(mean, std)
        x = torch.atanh(action)
        x = torch.max(torch.min(x, mean + 10 * std), mean - 10 * std)
        # x = torch.clamp(x, mean - 3 * std, mean + 3 * std)
        log_prob = gaussianDist.log_prob(x).sum(1, keepdim=True)
        log_prob -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = gaussianDist.entropy().sum(1, keepdim=True)
        return log_prob.exp(), entropy.mean()

    def update(self, Agent):
        """
        parameter들을 update한다.
        """
        self.actor.updateParameter(Agent.actor, tau=1.0)
    
    def loadParameters(self):
        """
        save file로 부터 load할 때 필요한 method이다.
        """
        self.actor.loadParameters()
        self.critic.loadParameters()
    
    def decayingLogStd(self, step):
        """
        logstd를 decaying한다.
        """
        if step < self.annealingStep:
            self.logStd -= self.deltaStd
    
    def clear(self, index):
        self.actor.clear(index)
        self.critic.clear(index)


class Node(nn.Module):
    """
        node can diverge output.
        also, collect input.

        To do this, node must specify the previous nodes and future nodes.
        the id of node is a kind of string name.
        the list of previous and future nodes is stored when initialized.

        Also, node have priority used for controlling the forwarding flow.

        To skip the priority, node mush store the current output.

        Store the Inputs for waiting the other inputs.
        
    """

    def __init__(
        self,
        data: dict
    ):
        super(Node, self).__init__()

        self.previousNodes: list
        self.previousNodes = []
        self.priority = data['prior']
        self.storedInput = []
        self.storedOutput = []
        self.data = data
    
    def setPrevNodes(self, prevNodes):
        prevNodes: list
        self.previousNodes = prevNodes
    
    def buildModel(self) -> None:
        self.model = constructNet(self.data)
    
    def clear_savedOutput(self) -> None:
        del self.storedOutput
        del self.storedInput
        self.storedInput = []
        self.storedOutput = []
    
    def addInput(self, _input) -> None:
        self.storedInput.append(_input)
    
    def getStoreOutput(self):
        return self.storedOutput

    def step(self) -> torch.tensor:
        if len(self.previousNodes) != 0:
            for prevNode in self.previousNodes:
                for prevInput in prevNode.storedOutput:
                    self.storedInput.append(prevInput)
        
        self.storedInput = tuple(self.storedInput)
        output = self.model.forward(self.storedInput)
        self.storedOutput.append(output.clone())
        self.storedOutput: torch.tensor
        return output


class AgentV2(nn.Module):
    """
        the priorityModel consists of priority and node.
        
    """

    def __init__(
        self,
        mData: dict,
        LSTMName=None
    ):
        super(AgentV2, self).__init__()
        """
            1. parsing the data
            2. load the weight
            3. return the list of weight for optimizer
            4. calculate the norm of gradient
            5. attach the device to layer
            6. forwarding!!
        """
        # data
        self.mData = mData

        # name
        self.moduleNames = list(self.mData.keys())
        
        # sorting the module layer
        self.moduleNames.sort()

        self.priorityModel, self.outputModelName, self.inputModelName = self.buildModel()
        self.priority = list(self.priorityModel.keys())
        self.priority.sort()
        self.LSTMname = LSTMName
        
    def buildModel(self) -> tuple:
        priorityModel = {}
        """
            key : priority
            element : dict,
                    key name
                    element node
        """

        outputModelName = []
        """
            element: list
                    prior, module name
        """

        inputModelName = {}
        """
            key : num of input
            element : list
                    element: [priority, name]
        """

        name2prior = {}
        """
            key : name of layer
            element: prior
        """

        for name in self.moduleNames:
            data = self.mData[name]
            data: dict
            name2prior[name] = data["prior"]
            if data["prior"] in priorityModel.keys():
                priorityModel[data["prior"]][name] = Node(data)
                priorityModel[data["prior"]][name].buildModel()
            else:
                priorityModel[data["prior"]] = {name: Node(data)}
                priorityModel[data["prior"]][name].buildModel()
            setattr(self, name, priorityModel[data["prior"]][name])
            
            if "output" in data.keys():
                if data["output"]:
                    outputModelName.append([data["prior"], name])
            
            if "input" in data.keys():
                for i in data["input"]:
                    if i in inputModelName.keys():
                        inputModelName[i].append([data["prior"], name])
                    else:
                        inputModelName[i] = [[data["prior"], name]]
        
        for prior in priorityModel.keys():
            node_dict = priorityModel[prior]
            for index in node_dict.keys():
                node = node_dict[index]
                if "prevNodeNames" in node.data.keys():
                    prevNodeNames = node.data["prevNodeNames"]
                    prevNodeNames: list
                    prevNodes = []
                    for name in prevNodeNames:
                        data = self.mData[name]
                        prevNodes.append(priorityModel[data["prior"]][name])
                    node.setPrevNodes(prevNodes)
        self.name2prior = name2prior
                        
        return priorityModel, outputModelName, inputModelName

    def loadParameters(self) -> None:
        pass

    def buildOptim(self) -> tuple:
        listLayer = []
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                listLayer.append(layerDict[name])
        
        return tuple(listLayer)

    def updateParameter(self, Agent, tau) -> None:
        """
        tau = 0, no change
        """
        Agent: AgentV2
        tau: float

        with torch.no_grad():
            for prior in self.priority:
                layerDict = self.priorityModel[prior]
                for name in layerDict.keys():
                    parameters = layerDict[name].parameters()
                    tParameters = Agent.priorityModel[prior][name].parameters()
                    for p, tp in zip(parameters, tParameters):
                        p.copy_((1 - tau) * p + tau * tp)

    def calculateNorm(self) -> float:
        totalNorm = 0
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                parameters = layerDict[name].parameters()
                for p in parameters:
                    norm = p.grad.data.norm(2)
                    totalNorm += norm

    def clippingNorm(self, maxNorm):
        inputD = []
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                inputD += list(layerDict[name].parameters())
        
        torch.nn.utils.clip_grad_norm_(inputD, maxNorm)
    
    def getCellState(self):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            return self.priorityModel[prior][self.LSTMname].getCellState()
    
    def setCellState(self, cellstate):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            self.priorityModel[prior][self.LSTMname].setCellState(cellstate)

    def zeroCellState(self):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            self.priorityModel[prior][self.LSTMname].zeroCellState()
    
    def zeroCellStateAgent(self, idx):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            self.priorityModel[prior][self.LSTMname].zeroCellStateAgent(idx)
    
    def detachCellState(self):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            self.priorityModel[prior][self.LSTMname].detachCellState()

    def to(self, device) -> None:
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                node = layerDict[name]
                node.to(device)
    
    def clear(self, index, step=0):
        if self.LSTMname is not None:
            prior = self.name2prior[self.LSTMname]
            self.priorityModel[prior][self.LSTMname].clear(index, step)

    def clear_savedOutput(self):

        for i in self.priority:
            nodeDict = self.priorityModel[i]
            for name in nodeDict.keys():
                node = nodeDict[name]
                node.clear_savedOutput()

    def forward(self, inputs) -> tuple:
        inputs: tuple
        
        for i, _input in enumerate(inputs):
            priorityName_InputModel = self.inputModelName[i]
            priorityName_InputModel: list
            for inputinfo in priorityName_InputModel:
                self.priorityModel[inputinfo[0]][inputinfo[1]].addInput(_input)

        for prior in range(self.priority[-1] + 1):
            for nodeName in self.priorityModel[prior].keys():
                node: str
                self.priorityModel[prior][nodeName].step()
        
        output = []
        for outinfo in self.outputModelName:
            out = self.priorityModel[outinfo[0]][outinfo[1]].getStoreOutput()
            for o in out:
                output.append(o)
        
        output = tuple(output)
        self.clear_savedOutput()
        return output