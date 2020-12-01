import torch
import numpy as np
import json
import torchvision.transforms.functional as TF

from baseline.baseNetwork import MLP, CNET, LSTMNET, CNN1D, Res1D, Cat


def showLidarImg(img):
    img = torch.tensor(img).float()
    img = TF.to_pil_image(img)
    img.show()
    

def calGlobalNorm(agent):
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
    
    keyList = list(optimData.keys())

    if 'name' in keyList:
        name = optimData['name']
        lr = optimData['lr']
        decay = 0 if 'decay' not in keyList else optimData['decay']
        eps = 1e-5 if 'eps' not in keyList else optimData['eps']
        clipping = False if 'clipping' not in keyList else optimData['clipping']
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


def constructNet(netData, iSize=1, WH=-1):
    netCat = netData['netCat']
    if netCat == 'Input':
        return None
    Net = [MLP, CNET, LSTMNET, CNN1D, Res1D, Cat]
    netName = ["MLP", "CNET", "LSTMNET", "CNN1D", "Res1D", "Cat"]
    ind = netName.index(netCat)

    baseNet = Net[ind]

    network = baseNet(
        netData
    )

    return network


class jsonParser:

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