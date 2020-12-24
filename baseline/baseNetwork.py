import torch
import torch.nn as nn


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

    def __init__(self, netData):
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
            self.linear = netData['linear']
        self.BN = netData['BN']
        self.iSize = netData['iSize']
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
            x = layer.forward(x)
        return x


class CNET(nn.Module):

    def __init__(self, netData):
        super(CNET, self).__init__()

        self.netData = netData
        self.iSize = netData['iSize']
        keyList = list(netData.keys())

        if "BN" in keyList:
            self.BN = netData['BN']
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
            self.linear = netData['linear']
        if self.linear:
            self.act[-1] = "linear"

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


class LSTMNET(nn.Module):

    def __init__(self, netData):
        super(LSTMNET, self).__init__()
        self.netData = netData
        self.hiddenSize = netData['hiddenSize']
        self.nLayer = netData['nLayer']
        iSize = netData['iSize']
        device = netData['device']
        self.device = torch.device(device)
        self.nAgent = self.netData['Number_Agent']
        self.CellState = (torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device), 
                          torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device))
        self.rnn = nn.LSTM(iSize, self.hiddenSize, self.nLayer)
        self.FlattenMode = netData['FlattenMode']
    
    def clear(self, index, step=0):
        hn, cn = self.CellState
        hn[step, index, :] = torch.zeros(self.hiddenSize).to(self.device)
        cn[step, index, :] = torch.zeros(self.hiddenSize).to(self.device)
        self.CellState = (hn, cn)
    
    def getCellState(self):
        return self.CellState

    def setCellState(self, cellState):
        self.CellState = cellState
    
    def detachCellState(self):
        self.CellState = (self.CellState[0].detach(), self.CellState[1].detach())

    def zeroCellState(self):
        self.CellState = (torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device), 
                          torch.zeros(1, self.nAgent, self.hiddenSize).to(self.device))
    
    def zeroCellStateAgent(self, idx):
        h = torch.zeros(1, 1, self.netData['hiddenSize'])
        c = torch.zeros(1, 1, self.netData['hiddenSize'])
        self.CellState[0][idx] = h
        self.CellState[1][idx] = c

    def forward(self, state):
        nDim = state.shape[0]
        if nDim == 1:
            output, (hn, cn) = self.rnn(state, self.CellState)
            if self.FlattenMode:
                output = torch.squeeze(output, dim=0)
            self.CellState = (hn, cn)
        else:
            output, (hn, cn) = self.rnn(state, self.CellState)
            if self.FlattenMode:
                output = output.view(-1, self.hiddenSize)
                output = output.view(-1, self.hiddenSize)
            self.CellState = (hn, cn)
        
        # output consists of output, hidden, cell state
        return output


class CNN1D(nn.Module):

    def __init__(
        self,
        netData,
    ):
        super(CNN1D, self).__init__()
        self.netData = netData
        self.iSize = netData['iSize']
        keyList = list(netData.keys())

        if "BN" in keyList:
            self.BN = netData['BN']
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
            self.linear = netData['linear']
        if self.linear:
            self.act.append("linear")
        self.buildModel()
    
    def buildModel(self):
        iSize = self.iSize
        mode = True
        for i, fSize in enumerate(self.fSize):
            if fSize == -1:
                mode = False
            if mode:
                self.add_module(
                    "conv1D_"+str(i+1),
                    nn.Conv1d(
                        iSize,
                        self.nUnit[i],
                        fSize,
                        stride=self.stride[i],
                        padding=self.padding[i],
                        bias=False)
                )
                iSize = self.nUnit[i]
            elif fSize == -1:
                self.add_module(
                    "Flatten",
                    nn.Flatten())
                # iSize = self.getSize()
            else:
                self.add_module(
                    "MLP_"+str(i+1),
                    nn.Linear(iSize, fSize, bias=False)
                )
                iSize = fSize
            
            act = getActivation(self.act[i])
            if act is not None and fSize != -1:
                self.add_module(
                    "act_"+str(i+1),
                    act
                )
        
    def getSize(self):
        ze = torch.zeros((1, self.iSize, self.L))
        k = self.forward(ze)
        k = k.view((1, -1))
        size = k.shape[-1]
        return size
    
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


def conv1D(
    in_num,
    out_num,
    kernel_size=3, padding=1, stride=1, 
    eps=1e-5, momentum=0.1,  
    is_linear=False, 
    is_batch=False
):
    
    if is_linear:
        temp = nn.Sequential(nn.Conv1d(in_num, out_num, kernel_size=kernel_size, 
                             padding=padding, stride=stride, bias=False))
    else:
        
        if is_batch:
            temp = nn.Sequential(
                nn.Conv1d(in_num, out_num, kernel_size=kernel_size, 
                          padding=padding, stride=stride, bias=False),
                nn.BatchNorm1d(out_num, eps=eps, momentum=momentum),
                nn.ReLU()
                )
        else:
            temp = nn.Sequential(
                nn.Conv1d(in_num, out_num, kernel_size=kernel_size, 
                          padding=padding, stride=stride, bias=False),
                nn.ReLU()
                )

    return temp


class ResidualConv(nn.Module):
    
    def __init__(self, in_num):
        super(ResidualConv, self).__init__()

        mid_num = int(in_num / 2)

        self.layer1 = conv1D(in_num, mid_num, kernel_size=1, padding=0)
        self.layer2 = conv1D(mid_num, in_num)

    def forward(self, x):
        
        residual = x
        
        out = self.layer1(x)
        z = self.layer2(out)

        z += residual

        return z


class Res1D(nn.Module):

    def __init__(
        self,
        aData
    ):
        super(Res1D, self).__init__()

        self.aData = aData
        self.iSize = self.aData['iSize']
        self.nBlock = self.aData['nBlock']
        self.isLinear = self.aData['linear']
        
        self.Model = self.buildModel()
        self.conv = conv1D(1, self.iSize)
    
    def buildModel(self):
        iSize = self.iSize
        layers = []

        for i in range(self.nBlock):
            layers.append(ResidualConv(iSize))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        
        y = self.conv(x)
        z = self.Model.forward(y)
        if self.isLinear:
            batchSize = z.shape[0]
            z = z.view((batchSize, -1))
        
        return z


class Cat(nn.Module):

    def __init__(self, data):
        super(Cat, self).__init__()
    
    def forward(self, x):
        return torch.cat(x, dim=-1)


class Unsequeeze(nn.Module):
    
    def __init__(
        self,
        data
    ):
        super(Unsequeeze, self).__init__()
        self.dim = data['dim']

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.dim)


class View(nn.Module):
    def __init__(
        self,
        data
    ):
        super(View, self).__init__()
        self.shape = data['shape']

    def forward(self, x):
        return x.view(self.shape)
