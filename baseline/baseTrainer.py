import gym
import torch
import datetime
import numpy as np

from baseline.utils import jsonParser
from mlagents_envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class OFFPolicy:

    def __init__(self, fName):
        parser = jsonParser(fName)
        self.data = parser.loadParser()
        self.aData = parser.loadAgentParser()
        self.optimData = parser.loadOptParser()

        keyList = list(self.data.keys())
        torch.manual_seed(self.data['seed'])

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

            self.brain = self.env.brain_names[0]
        else:
            self.env = gym.make(name)
            self.evalEnv = gym.make(name)
        
        self.nReplayMemory = int(self.data['nReplayMemory'])
        self.gamma = self.data['gamma']
        self.bSize = self.data['bSize']
        self.rScaling = self.data['rScaling']
        self.lrFreq = self.data['lrFreq']

        self.runStep = self.data['runStep']
        self.startStep = self.data['startStep']
        self.episodeP = self.data['episodeP']

        self.sPath = self.data['sPath']
        self.writeTMode = \
            True if 'writeTMode' not in keyList else self.data['writeTMode'] == "True"
        if self.writeTMode:
            time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
            self.tPath = self.data['tPath'] + self.data['envName']+time
            self.writer = SummaryWriter(self.tPath)
        self.lPath = self.data['lPath']
        self.device = self.data['device']
        self.inferMode = self.data['inferMode'] == "True"
        self.renderMode = self.data['renderMode'] == "True"
        
        self.keyList = keyList
        self.aSize = self.data['aSize']
        self.sSize = self.data['sSize']
        self.nAgent = self.data['nAgent']
        self.evalObsSet = deque(maxlen=self.sSize[0])
        self.best = self.data['best']
        self.evalP = self.data['evalP']
        assert ~(self.uMode is False and self.nAgent > 1), "nAgent must be 1,"

    def reset(self):
        pass

    def genOptim(self):
        pass

    def initializePolicy(self):
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

    def lrSheduling(self):
        pass

    def eval(self):
        pass

    def writeTrainInfo(self):
        self.info = """
        envName:{}
        startStep:{:3d}
        nReplayMemory:{:5d}
        bSize:{:3d}
        lrFreq:{:3d}
        rScaling:{:3d}
        gamma:{}
        """.format(
            self.data['envName'], self.startStep,
            self.nReplayMemory, self.bSize, self.lrFreq,
            self.rScaling, self.gamma)
        optimKeyList = list(self.optimData.keys())
        for key in optimKeyList[:3]:
            self.info += """
        optim_{}:
            """.format(key)
            optimList = list(self.optimData[key].keys())
            for data in optimList:
                self.info += """
                {}:{}
                """.format(data, self.optimData[key][data])