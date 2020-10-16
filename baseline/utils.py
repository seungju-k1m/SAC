import torch
import numpy as np
import json


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


def getOptim(optimData, agent):
    
    keyList = list(optimData.keys())

    if 'name' in keyList:
        name = optimData['name']
        lr = optimData['lr']
        decay = 0 if 'decay' not in keyList else optimData['decay']
        eps = 1e-5 if 'eps' not in keyList else optimData['eps']
        
        if name == 'adam':
            optim = torch.optim.Adam(
                agent.parameters(),
                lr=lr,
                weight_decay=decay,
                eps=eps
                )
        if name == 'sgd':
            momentum = 0 if 'momentum' not in keyList else optimData['momentum']

            optim = torch.optim.SGD(
                agent.parameters(),
                lr=lr,
                weight_decay=decay,
                momentum=momentum
            )
        if name == 'rmsprop':
            optim = torch.optim.RMSprop(
                agent.parameters(),
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