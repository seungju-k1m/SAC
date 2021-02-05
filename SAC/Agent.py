import torch
import torch.nn as nn
from baseline.utils import constructNet
from baseline.baseNetwork import baseAgent, LSTMNET


class sacAgent(baseAgent):

    def __init__(
        self, 
        aData
    ):
        super(sacAgent, self).__init__()
        self.aData = aData
        self.keyList = list(self.aData.keys())
        device = self.aData['device']
        self.device = torch.device(device)
        self.buildModel()
        self.to(self.device)
    
    def buildModel(self):
        for netName in self.keyList:
            if netName == "actor":
                netData = self.aData[netName]
                self.actor = AgentV1(netData)

            if netName == "critic":
                netData = self.aData[netName]
                self.critic01 = AgentV1(netData)
                self.critic02 = AgentV1(netData)
            
            if self.ICMMode:
                if netName == "Forward":
                    netData = self.aData[netName]
                    self.ForwardM = AgentV1(netData)
                
                if netName == "inverseModel":
                    netData = self.aData[netName]
                    self.inverseM = AgentV1(netData)
                
                if netName == "Feature":
                    netData = self.aData[netName]
                    self.FeatureM = AgentV1(netData)

        self.temperature = torch.zeros(1, requires_grad=True, device=self.aData['device'])
    
    def to(self, device):
        self.actor.to(device)
        self.critic01.to(device)
        self.critic02.to(device)
        if self.ICMMode:
            self.FeatureM.to(device)
            self.inverseM.to(device)
            self.ForwardM.to(device)

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

        critic01, critic02 = self.criticForward(state, action)

        return action, logProb, (critic01, critic02), entropy

    def actorForward(self, state, dMode=False):

        output = self.actor.forward(state)[0]
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
        
        critic01 = self.critic01.forward(state)[0]
        critic02 = self.critic02.forward(state)[0]
    
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

    def loadParameters(self):
        self.actor.loadParameters()
        self.critic01.loadParameters()
        self.critic02.loadParameters()


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
        mData: dict
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

        for name in self.moduleNames:
            data = self.mData[name]
            data: dict
            if data["prior"] in priorityModel.keys():
                priorityModel[data["prior"]][name] = Node(data)
                priorityModel[data["prior"]][name].buildModel()
            else:
                priorityModel[data["prior"]] = {name: Node(data)}
                priorityModel[data["prior"]][name].buildModel()
            
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
                        
        return priorityModel, outputModelName, inputModelName

    def loadParameters(self) -> None:
        pass

    def buildOptim(self) -> None:
        pass

    def updateParameter(self) -> None:
        pass

    def calculateNorm(self) -> None:
        pass

    def to(self, device) -> None:
        pass

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
          
            
class AgentV1(nn.Module):

    def __init__(
        self,
        mData
    ):
        super(AgentV1, self).__init__()
        self.mData = mData
        self.moduleNames = list(self.mData.keys())
        self.moduleNames.sort()
        self.buildModel()
    
    def buildModel(self):
        self.model = {}
        self.listModel = []
        self.connect = {}
        for _ in range(10):
            self.connect[_] = []
        
        for i, name in enumerate(self.moduleNames):
            self.model[name] = constructNet(self.mData[name])
            setattr(self, '_'+str(i), self.model[name])
            if 'input' in self.mData[name].keys():
                inputs = self.mData[name]['input']
                for j in inputs:
                    self.connect[j].append(name)
    
    def loadParameters(self):
        for i, name in enumerate(self.moduleNames):
            self.model[name] = getattr(self, '_'+str(i))
    
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
