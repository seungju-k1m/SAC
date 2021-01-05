import torch
import numpy as np
import json
import torchvision.transforms.functional as TF

from baseline.baseNetwork import MLP, CNET, LSTMNET, CNN1D, Res1D, Cat, Unsequeeze, View


"""
utils의 경우, 다양한 상황에서 사용되는 기타 method이다.
"""


def showLidarImg(img):
    """
    args:
        img:np.array, [C, H, W]
    """
    img = torch.tensor(img).float()
    img = TF.to_pil_image(img)
    img.show()
    

def calGlobalNorm(agent):
    """
    agent의 graident의 norm/sum을 구한다.
    """
    totalNorm = 0
    for p in agent.parameters():
        norm = p.grad.data.norm(2)
        totalNorm += norm
    return totalNorm


def clipByGN(agent, maxNorm):
    totalNorm = calGlobalNorm(agent)
    for p in agent.parameters():
        factor = maxNorm/np.maximum(totalNorm, maxNorm)
        p.grad *= factor


def getOptim(optimData, agent, floatV=False):
    
    """
    configuration에서 정의된 optimizer setting을 지원한다.

    args:
        optimData:
            name:[str] optimizer의 이름
            lr:[float] learning rate
            decay:[float] decaying(L2 Regularzation)
            eps:[float], 
            clipping:deprecated
        agent:[tuple, torch.nn], 해당 optimizer가 담당하는 weight들 Agentv1.buildOptim을 통해서 호출
        floatV:[bool], weight이 torch.nn이 아니라 tensor인 경우
            
    """
    
    keyList = list(optimData.keys())

    if 'name' in keyList:
        name = optimData['name']
        lr = optimData['lr']
        decay = 0 if 'decay' not in keyList else optimData['decay']
        eps = 1e-5 if 'eps' not in keyList else optimData['eps']
        
        if floatV:
            inputD = agent
        elif type(agent) == tuple:
            inputD = []
            for a in agent:
                inputD += list(a.parameters())

        else:
            inputD = agent.parameters()
        if name == 'adam':
            optim = torch.optim.Adam(
                inputD,
                lr=lr,
                weight_decay=decay,
                eps=eps
                )
        if name == 'sgd':
            momentum = 0 if 'momentum' not in keyList else optimData['momentum']

            optim = torch.optim.SGD(
                inputD,
                lr=lr,
                weight_decay=decay,
                momentum=momentum
            )
        if name == 'rmsprop':
            optim = torch.optim.RMSprop(
                inputD,
                lr=lr,
                weight_decay=decay,
                eps=eps
            )
    
    return optim


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


def constructNet(netData):
    """
    configuration에 따라 해당 network를 반환
    
    args:
        netData:dict
    """
    netCat = netData['netCat']
    if netCat == 'Input':
        return None
    Net = [MLP, CNET, LSTMNET, CNN1D, Res1D, Cat, Unsequeeze, View]
    netName = ["MLP", "CNET", "LSTMNET", "CNN1D", "Res1D", "Cat", "Unsequeeze", "View"]
    ind = netName.index(netCat)

    baseNet = Net[ind]
    network = baseNet(
        netData
    )

    return network


class jsonParser:
    """
    configuration은 *.json 형태이기 때문에
    이를 dictionary형태로 변환시켜주는 class
    """

    def __init__(self, fileName):
        with open(fileName) as jsonFile:
            self.jsonFile = json.load(jsonFile)
    
    def loadParser(self):
        return self.jsonFile
    
    def loadAgentParser(self):
        agentData = self.jsonFile.get('agent')
        agentData['sSize'] = self.jsonFile['sSize']
        agentData['aSize'] = self.jsonFile['aSize']
        agentData['device'] = self.jsonFile['device']
        agentData['gamma'] = self.jsonFile['gamma']
        return agentData
    
    def loadOptParser(self):
        return self.jsonFile.get('optim')


class PidPolicy:
    """
    PID Policy를 위한 class
    """
    def __init__(self, parm):
        self.parm = parm

    def pid_policy(self, dx, dy, yaw):
        """
        PID policy in safe situation
        """
        self.dx = dx
        self.dy = dy
        self.yaw = yaw

        e_s, e_yaw = self.calculate_e()
        uv_pid = self.parm['Kp_lin'] * e_s
        uw_pid = self.parm['Kp_ang'] * e_yaw

        uv_pid = np.clip(uv_pid, self.parm['uv_min'], self.parm['uv_max'])
        uw_pid = np.clip(uw_pid, self.parm['uw_min'], self.parm['uw_max'])

        return uv_pid, uw_pid

    def calculate_e(self):
        """
        Calculate longitudinal and lateral error for PID policy
        """
        e_s = np.sqrt(np.power(self.dx, 2) + np.power(self.dy, 2)) * np.cos(np.arctan2(self.dy,self.dx) - self.yaw)
        yaw_ref = np.arctan2(self.dy, self.dx)
        e_yaw_ = yaw_ref - self.yaw
        e_yaw = np.arctan2(np.sin(e_yaw_), np.cos(e_yaw_))
        return e_s, e_yaw
