import torch
import torch.nn as nn

"""
baseNetwork module은 neural network를 만들기 위한 재료를 제공한다.
MLP, CNN, LSTM으로 대표되는 신경망뿐만 아니라, 
neural network의 forwarding을 위한 전처리 과정으로, Concat, view, Unsequeeze등을 지원한다.
"""


def getActivation(actName, **kwargs):
    """
    다양한 activation function을 지원하는 method.
    args:
        actName:[str], activation의 이름
                <'relu', 'leakyRelu', 'sigmoid', 'tanh', 'linear'>지원
        kwargs:
            slope:[float], leakyRelu의 경우, slope를 지정할 수 있다.default:1e-2 
    output:
        act:[torch.nn.activation]
    """
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
    """
    baseAgent는 agent의 기초적인 method를 선언해주는 class이다.
    일종의 abstraction이다.
    """

    def __init__(self):
        super(baseAgent, self).__init__()

    def buildModel(self):
        pass

    def calLoss(self):
        pass
    
    def forward(self):
        pass
    

class MLP(nn.Module):
    """
    MLP class는 multi layer perceptron을 지원한다.
    MLP는 다음을 통해 설정할 수 있다.
    configuration:
        args:
            iSize:[int], input의 형태
            nLayer:[int], layer의 갯수
            fSize:[list, int], layer당 unit의 갯수, 이때 len(fSize) == nLayer여야한다.
            act:[list, str], layer에 적용되는 activation function
        kwargs:
            BN:Batch normalization, default:false
    """

    def __init__(self, netData):
        super(MLP, self).__init__()
        self.netData = netData
        self.nLayer = netData['nLayer']
        self.fSize = netData['fSize']
        act = netData['act']
        if not isinstance(act, list):
            act = [act for i in range(self.nLayer)]
        self.act = act
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
                if (i == (self.nLayer-1)) is not True:
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
    """
    CNET class는 CNN을 지원한다.
    CNN는 다음을 통해 설정할 수 있다.
    configuration:
        args:
            iSize:[int], input의 channel
            nLayer:[int], layer의 갯수
            fSize:[list, int], layer에 적용되는 kernel의 크기
            nUnit:[list, int], layer가 가지고 있는 unit 갯수.
            padding:[list, int], padding
            stride:[list, int], stride
            act:[list, str], layer에 적용되는 activation function
        kwargs:
            BN:Batch normalization, default:false
    """

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
                if (i == (self.nLayer-1)) is not True:
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

    def getSize(self, WH=96):
        """
        CNNs의 output의 크기를 확인할 수 있다.
        args:
            WH:[int], input의 width, height, default:96
        """
        ze = torch.zeros((1, self.iSize, WH, WH))
        k = self.forward(ze)
        k = k.view((1, -1))
        size = k.shape[-1]
        return size

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class LSTMNET(nn.Module):
    """
    LSTMNET class는 LSTM을 지원한다.
    LSTM는 다음을 통해 설정할 수 있다.
    configuration:
        args:
            iSize:[int], input의 형태
            nLayer:[int], layer의 갯수, 현재 1밖에 지원안함!
            hiddenSize:[int], cell state의 크기
            Number_Agent:[int], 현재 환경에서 돌아가는 agent의 수
            FlattenMode:[bool], lstm의 output은 <seq, batch, hidden>를 
                                                <seq*batch, hidden>로 변환
    """

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
        """
        deprecated by zeroCellStateAgent
        """
        hn, cn = self.CellState
        hn[step, index, :] = 0
        cn[step, index, :] = 0
        self.CellState = (hn, cn)
    
    def getCellState(self):
        """
        CellState을 반환한다.
        output:
            dtype:tuple, (hstate, cstate)
            state:torch.tensor, shape:[1, Agent의 숫자, hiddenSize]
        """
        return self.CellState

    def setCellState(self, cellState):
        """
        CellState를 설정한다.
        args:
            cellState:tuple, (hstate, cstate)
            state:torch.tensor, shape:[1, Agent의 숫자, hiddenSize]
        """
        self.CellState = cellState
    
    def detachCellState(self):
        "LSTM의 BTTT를 지원하기 위해서는 detaching이 필요하다."
        # self.CellState = (self.CellState[0].detach(), self.CellState[1].detach())
        self.CellState[0].detach_()
        self.CellState[1].detach_()

    def zeroCellState(self):
        """
        cellState를 zero로 변환하는 과정이다.
        환경이 초기화 되면, lstm역시 초기화 되어야한다.
        """
        self.CellState[0][:, :, :] = 0
        self.CellState[1][:, :, :] = 0
    
    def zeroCellStateAgent(self, idx):
        """
        모든 agent가 아닌 특정 agent의 cell state를 0으로 반환
        """
        h = torch.zeros(self.netData['hiddenSize'])
        c = torch.zeros(self.netData['hiddenSize'])
        self.CellState[0][0, idx] = h
        self.CellState[1][0, idx] = c

    def forward(self, state):
        nDim = state.shape[0]
        if nDim == 1:
            output, self.CellState = self.rnn(state, self.CellState)
            if self.FlattenMode:
                output = torch.squeeze(output, dim=0)
        else:
            output, self.CellState = self.rnn(state, self.CellState)
            if self.FlattenMode:
                output = output.view(-1, self.hiddenSize)
                output = output.view(-1, self.hiddenSize)
        
        # output consists of output, hidden, cell state
        return output


class CNN1D(nn.Module):
    """
    CNN1D class는 CNN1D을 지원한다.
    CNN1D는 다음을 통해 설정할 수 있다.
    configuration:
        args:
            iSize:[int], input의 channel
            nLayer:[int], layer의 갯수
            fSize:[list, int], layer에 적용되는 kernel의 크기
            nUnit:[list, int], layer가 가지고 있는 unit 갯수.
            padding:[list, int], padding
            stride:[list, int], stride
            act:[list, str], layer에 적용되는 activation function
            linear:[bool], batch, channel, w -> batch, channel * w
        kwargs:
            BN:Batch normalization, default:false

    """

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
        
    def getSize(self, WH):
        """
        CNNs의 output의 크기를 확인할 수 있다.
        args:
            WH:[int], input의 width, height, default:96
        """
        ze = torch.zeros((1, self.iSize, WH))
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


class Cat(nn.Module):
    """
    concat을 지원한다.
    """
    
    def __init__(self, data):
        super(Cat, self).__init__()
    
    def forward(self, x):
        return torch.cat(x, dim=-1)


class Unsequeeze(nn.Module):
    """
    unsequeeze를 지원
    """
    
    def __init__(
        self,
        data
    ):
        super(Unsequeeze, self).__init__()
        self.dim = data['dim']

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.dim)


class View(nn.Module):
    """
    view를 지원. 이때 view는 shape를 변환하는 것을 의미한다.
    """
    def __init__(
        self,
        data
    ):
        super(View, self).__init__()
        self.shape = data['shape']

    def forward(self, x):
        return x.view(self.shape)


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
