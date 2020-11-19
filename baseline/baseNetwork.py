import math
import torch

import numpy as np
import torch.nn as nn


class Branches:
    """
        Linear Neural Network Module
        Most Neural Network structure can be represeted by linearity.
        junction points such as diverging point and parallel forwarding point can cause the synchronized problem.

        To handle this problem, I unity the two core algorithm, Go and Stop and recursion.
        I divide the whole network by two term, Go and Stop.
        Go and Stop do commonly have their own priority and To priority.
        Each priority can be interpreted the address of each module and the objective address,

        recursively forwarding at each start point. then they end up some 'STOP' object
        some cases when the module need the other input/feature that yet get there, the forwarding procedure
        must be stopped until the other feature come there.
        If specific conditions are satisfied, then they act like GO modules.
    """
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        self.keys = cfg.keys()
        self.build()
        self.MAXPRIORITY = -1
    
    def build(self):
        self.Branch = {}
        self.prior = {}
        self.stackPrior = []
        for key in self.keys():
            if 'Branch' in key: 
                self.Branch[key] = Branch(self.cfg[key])
                prior = self.Branch[key].prior
                if self.Branch[key].stop:
                    self.stackPrior.append(key)
                if prior not in self.prior:
                    self.prior[prior] = []
                    self.prior[prior].append(self.Branch[key])
            
            if prior > self.MAXPRIORITY:
                self.MAXPRIORITY = prior

    def flow(self, inFeatures, prior):

        modules = self.prior[prior]
        for inFeature in inFeatures:
            for module in modules:
                nPriors = module.To
                for nP in nPriors:
                    if module.stop and module.cond is False:
                        module.stack(inFeature)
                    else:
                        inFeature = (inFeature)
                        outFeature = module.forward(inFeature)
                        self.flow(outFeature, nP)

    def forward(self, inputs, lstmInputs=None):

        if lstmInputs is not None:
            self.flow(lstmInputs, prior=-2)
        for input in inputs:
            self.flow(input, prior=-1)

        LastModule = self.prior[self.MAXPRIORITY]
        output = LastModule.forward()
        return output


class Branch:
    """
    """
    def __init__(
        self, 
        cfg,
        prior,
        To=None,
        stop=False,
        Mux=False,
        deMux=False,
        **kwargs
    ):
        self.cfg = cfg
        self.prior = prior
        if To is None:
            self.To = [prior + 1]
        else:
            self.To = To
        self.stop = stop
        self.cond = True
        if self.stop:
            self.cell = []
            self.cond = False
            self.nInput = self.cfg["nInput"]
        
        self.Mux = Mux
        if self.Mux:
            self.nMux = kwargs['nMux']
    
    def stack(self, inFeature):
        if self.stop and self.cond is False:
            self.cell.append(inFeature)
            if len(self.cell) == self.nInput:
                self.cond = True

    def forward(self, inFeature=None):
        pass
            

def getActivation(actName, **kwargs):
    if actName == 'relu':
        act = torch.nn.ReLU()
    if actName == 'leakyRelu':
        nSlope = 1e-2 if 'slope' not in kwargs.keys() else kwargs['slope']
        act = torch.nn.LeakyReLU(negative_slope=nSlope)
    if actName == 'sigmoid':
        act = torch.nn.Sigmoid()
    if actName == 'tanh':
        act = torch.nn.Tanh()
    if actName == 'linear':
        act = None
    
    return act


class baseModule(nn.Module):
    def __init__(self):
        super(baseModule, self).__init__()
        self.prevNet = None
        self.nextNet = None
        self.input = False


class baseAgent(nn.Module):

    def __init__(self):
        super(baseAgent, self).__init__()

    def buildModel(self):
        pass

    def calLoss(self):
        pass
    
    def forward(self):
        pass


class MLP(baseModule):

    def __init__(self, netData, iSize=1):
        super(MLP, self).__init__()
        self.netData = netData
        self.nLayer = netData['nLayer']
        self.fSize = netData['fSize']
        act = netData['act']
        if not isinstance(act, list):
            act = [act for i in range(self.nLayer)]
        self.act = act
        if 'linear' not in netData.keys():
            self.linear = False
        else:
            self.linear = netData['linear'] == "True"
        self.BN = netData['BN'] == "True"
        self.iSize = iSize
        self.buildModel()

    def buildModel(self):
        iSize = self.iSize
        for i in range(self.nLayer):
            self.add_module(
                "MLP_"+str(i+1),
                nn.Linear(iSize, self.fSize[i], bias=False)
                )
            
            if self.BN:
                if (self.linear and i == (self.nLayer-1)) is not True:
                    self.add_module(
                        "batchNorm_"+str(i+1),
                        nn.BatchNorm1d(self.fSize[i])
                    )
            act = getActivation(self.act[i])
            if act is not None:
                self.add_module(
                    "act_"+str(i+1),
                    act
                )
            iSize = self.fSize[i]
    
    def forward(self, x, shortcut=None):
        for i, layer in enumerate(self.children()):
            x = layer(x)
        return x


class CNN2D(baseModule):

    def __init__(self, netData, iSize=[3, 96, 96]):
        super(CNN2D, self).__init__()

        self.netData = netData
        self.iSize = iSize
        keyList = list(netData.keys())
        self.WH = WH

        if "BN" in keyList:
            self.BN = netData['BN'] == "True"
        else:
            self.BN = False
        self.nLayer = netData['nLayer']
        self.fSize = netData['fSize']
        self.nUnit = netData['nUnit']
        self.padding = netData['padding']
        self.stride = netData['stride']
        act = netData['act']
        if not isinstance(act, list):
            act = [act for i in range(self.nLayer)]
        self.act = act
        if 'linear' not in netData.keys():
            self.linear = False
        else:
            self.linear = netData['linear'] == "True"
        if self.linear:
            act[-1] = "linear"

        self.buildModel()
        
    def buildModel(self):

        iSize = self.iSize
        mode = True
        for i, fSize in enumerate(self.fSize):
            if fSize == -1:
                mode = False
            if mode:
                self.add_module(
                    "conv_"+str(i+1),
                    nn.Conv2d(iSize, self.nUnit[i], fSize,
                              stride=self.stride[i],
                              padding=self.padding[i],
                              bias=False)
                )
                iSize = self.nUnit[i]
            elif fSize == -1:
                self.add_module(
                    "Flatten",
                    nn.Flatten())
                iSize = self.getSize()
            else:
                self.add_module(
                    "MLP_"+str(i+1),
                    nn.Linear(iSize, fSize, bias=False)
                )
                iSize = fSize
            if self.BN and fSize is not -1:
                if (self.linear and i == (self.nLayer-1)) is not True:
                    if mode:
                        self.add_module(
                            "batchNorm_"+str(i+1),
                            nn.BatchNorm2d(self.nUnit[i])
                        )
                    else:
                        self.add_module(
                            "batchNorm_"+str(i+1),
                            nn.BatchNorm1d(fSize)
                        )
            act = getActivation(self.act[i])
            if act is not None and fSize != -1:
                self.add_module(
                    "act_"+str(i+1),
                    act
                )

    def getSize(self):
        ze = torch.zeros((1, self.iSize, self.WH, self.WH))
        k = self.forward(ze)
        k = k.view((1, -1))
        size = k.shape[-1]
        return size

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class LSTMNET(baseModule):

    def __init__(self, netData, iSize=1):
        super(LSTMNET, self).__init__()
        self.netData = netData
        self.hiddenSize = netData['hiddenSize']
        self.nLayer = netData['nLayer']
        self.rnn = nn.LSTM(iSize, self.hiddenSize, self.nLayer)

    def forward(self, state, lstmState):
        output, (hn, cn) = self.rnn(state, lstmState)
        # output consists of output, hidden, cell state
        return output, (hn, cn)


class Cat(baseModule):
    def __init__(self):
        super(Cat, self).__init__()
    
    def forward(self, x):
        f1, f2 = x
        output = torch.cat((f1, f2), dim=0)
        return output


class Flatten(baseModule):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        x = x.view((1, -1))
        return x


def getCNETDim(iSize, netData):
    _, H_in, W_in = iSize,
    stride = netData['stride']
    numLayer = len(stride)
    fSize = netData['fSize'][:numLayer]
    padding = netData['padding']
    C_out = netData['nUnit'][-1]

    for s, f, p in zip(stride, fSize, padding):
        H_in = math.floor((H_in + 2 * p - f - 2) / s + 1)
        W_in = math.floor((W_in + 2 * p - f - 2) / s + 1)
    
    return (C_out, H_in, W_in)


def getCNET1DDim(iSize, netData):
    _, L_in = iSize
    stride = netData['stride']
    numLayer = len(stride)
    fSize = netData['fSize'][:numLayer]
    padding = netData['padding']
    C_out = netData['nUnit'][-1]

    for s, f, p in zip(stride, fSize, padding):
        L_in = math.floor((L_in + 2 * p - f -2) / s + 1)
    
    return [C_out, L_in]


def getOutput(iSize, netData):
    netCat = netData["netCat"]
    cat = ['MLP', "CNET", "LSTMNET", "CNET1D", "Flatten", "Cat"]
    if netCat not in cat:
        raise RuntimeError("can not get the output size!!")
    if netCat != "Cat":
        iSize = iSize[0]

    if netCat == 'MLP':
        output = [netData['fSize'][-1]]
    elif netCat == 'CNET':
        C_out, H_out, W_out = getCNETDim(iSize)
        isFlatten = netData['fSize'][-1] == -1
        if isFlatten:
            output = [C_out * H_out * W_out]
        else:
            output = [C_out, H_out, W_out]
    elif netCat == "LSTMNET":
        output = [netData['hiddenSize']]
    elif netCat == "CNET1D":
        output = getCNET1DDim(iSize, netData)
    elif netCat == "Flatten":
        output = 1
        for i in iSize:
            output *= i
        output = [output]
    elif netCat == "Cat":
        output = 0
        for i in iSize:
            output += i[0]
        output = [output]
    else:
        raise RuntimeError("already error!!")

    return output


def constructNet(netData, iSize):
    netCat = netData['netCat']
    Net = [MLP, CNET, LSTMNET, Cat, Flatten]
    netName = ["MLP", "CNET", "LSTMNET", "Cat", "Flatten"]
    ind = netName.index(netCat)

    baseNet = Net[ind]

    return network


class Agent:
    """
    class Agent has the following members and methods

    members:
        aData:[dict]
            it has the named key such as Module0x. each module has the
            information about network
        iSize:
            the number of inputs can be more than 1.
    methods:
        to:
            set the device of the network
        getISize:
            Sometimes, the input of each network can be verified. using
            this method, we can get the input size.
        buildModel:
            build Model. the type of this method is dictionary.
            the keys of dicctionay has the common rule, Modulexx.
            and each module must be indicated and indicate the other network.
        forward:
            input can be 
    """
    def __init__(
        self,
        aData,
        iSize=None
    ):
        self.aData = aData
        self.iSize = iSize
        # set the dictionary for the modules
        self.Module = {}
        key = np.array(list(self.aData.keys()))
        mask = ["Module" in i for i in key]
        key = key[mask]
        key.sort()
        self.key = key
        self.getISize()
        # self.buildModel()

    def to(self, device):
        pass

    def getISize(self):
        self.inputInfo = {}
        self.dirInfo = {}
        for key in self.key:
            self.dirInfo[key] = {}
            self.inputInfo[key] = {}
        pInfo = [None]
        i = 0
        numBranch = 0
        length = len(self.key)
        for info in self.key:
            netData = self.aData[info]
            if "branch" in netData:
                branch = netData['branch'] == "True"
                if branch:
                    numBranch += 1
                    if numBranch % 2 == 1:
                        length -= 1
            else:
                branch = False

            if 'shortcut' in netData.keys():
                shortcut = netData['shortcut']
            else:
                shortcut = [-1]
            self.dirInfo[info]['backward'] = []
            for ss in shortcut:
                pModule = pInfo[ss]
                self.dirInfo[info]['backward'].append(pModule)
            if branch:
                pass
            else:
                i += 1
                pInfo.append(info)
            
            if "input" in netData.keys():
                self.dirInfo[info]["input"] = netData['input'] == "True"
        
        for info in self.key:
            dirInfo = self.dirInfo[info]
            netData = self.aData[info]
            iSize = []
            if "input" in dirInfo.keys():
                iSize.append(self.iSize[0])
                self.iSize.pop(0)
                if netData['netCat'] == "Cat":
                    prevNet = self.dirInfo[info]['backward'][0]
                    iSize.append(self.inputInfo[prevNet]['oSize'])
            else:
                prevNets = self.dirInfo[info]['backward']
                for net in prevNets:
                    iSize.append(self.inputInfo[net]['oSize'])
            self.inputInfo[info]["iSize"] = iSize
            oSize = getOutput(iSize, netData)
            self.inputInfo[info]['oSize'] = oSize
        print(1)
if __name__ == "__main__":
    aData = {
    "Module01": {
      "netCat":"CNET1D",
      "nLayer":1,
      "nUnit": [8],
      "stride": [2],
      "fSize": [7],
      "padding": [1],
      "act": ["relu"],
      "input":"True"
    },
    "Module02": {
      "netCat": "Flatten"
    },
    "Module03": {
      "netCat": "Cat",
      "count":2,
      "iSize":[-1, 6],
      "input": "True"
    },
    "Module04": {
      "netCat": "LSTMNET",
      "hiddenSize": 256
    },
    "Module05": {
      "netCat": "MLP",
      "nLayer": 2,
      "fSize": [256, 4],
      "act": ["relu", "linear"],
      "branch":"True"
    },
    "Module06": {
      "netCat": "MLP",
      "nLayer":2,
      "fSize":[256, 1],
      "act":["relu", "linear"],
      "branch":"True"
    }
    }
    iSize = [[1, 120], [6]]
    ag = Agent(aData, iSize)


            