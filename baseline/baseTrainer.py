import gym
import numpy as np

from baseline.utils import jsonParser
from mlagents_envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter


class OFFPolicy:

    def __init__(self, fName):
        parser = jsonParser(fName)
        self.data = parser.loadParser()
        self.aData = parser.loadAgentParser()
        self.optimData = parser.loadOptParser()

        keyList = list(parser.keys())

        self.uMode = False if 'unityEnv' not in keyList else self.data['unityEnv'] == "True"
        self.sMode = False if 'sMode' not in keyList else self.data['sMode'] == "True"

        if self.sMode:
            self.tau = self.data['tau']
        else:
            self.udateP = self.data['updateP']
        
        name = self.data['envName']
        print(name)
        if self.uMode:
            id_ = np.random.randint(10, 100, 1)[0]
            self.env = UnityEnvironment(name, worker_id=id_)
        else:
            self.env = gym.make(name)
        
        self.nReplayMemory = self.data['nReplayMemory']
        self.gamma = self.data['gamma']
        self.bSize = self.data['bSize']
        self.rScaling = self.data['rSCaling']
        self.lrFreq = self.data['lrFreq']

        self.runStep = self.data['runStep']
        self.startStep = self.data['startStep']
        self.episodeP = self.data['episodeP']

        self.sPath = self.data['sPath']
        self.writeTMode = \
            True if 'writeTMode' not in keyList else self.data['writeTMode'] == "True"
        if self.writeTMode:
            self.tPath = self.data['tPath']
            self.writer = SummaryWriter(self.tPath)
        self.lPath = self.data['lPath']
        self.device = self.data['device']
        self.inferMode = self.data['inferMode'] == "True"
        self.renderMode = self.data['renderMode'] == "True"

    def genNetwork(self):
        pass

    def genObsSets(self):
        pass

    def reset(self):
        pass

    def genOptim(self):
        pass

    def initializePolicy(self):
        self.genNetwork()
        self.genObsSets()
        self.reset()
        self.genOptim()

    def ppState(self, obs):
        pass

    def appendMemory(self, data):
        pass

    def getAction(self, state):
        pass

    def targetNetUpdate(self):
        pass

    def train(self):
        pass

    def run(self):
        pass


