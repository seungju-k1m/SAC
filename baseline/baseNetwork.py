import torch
import torch.nn as nn
from baseline.utils import getActivation


class baseAgent(nn.Module):

    def __init__(self):
        super(baseAgent, self).__init__()

    def buildModel(self):
        pass

    def calLoss(self):
        pass
    
    def forward(self):
        pass


class MLP(nn.Module):

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
        if shortcut is not None:
            lenNet = len(self.fSize) * 2 - 1
            shortcut = lenNet + shortcut
        for i, layer in enumerate(self.children()):
            x = layer(x)
            if shortcut is not None and i == shortcut:
                middleOutput = x
        if shortcut is not None:
            x = (middleOutput, x) 
        return x


class CNET(nn.Module):

    def __init__(self, netData, iSize=3, WH=80):
        super(CNET, self).__init__()

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
