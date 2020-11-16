import torch
import datetime
import numpy as np

from baseline.utils import jsonParser
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class ONPolicy:

    def __init__(self, cfg):
        # 1. load parser
        parser = jsonParser(cfg)
        self.data = parser.loadParser()
        self.aData = parser.loadAgentParser()
        self.optimData = parser.loadOptParser()

        keyList = list(self.data.keys())
        torch.manual_seed(self.data['seed'])

        # 2. hyper-parameter setting

        self.gamma = self.data['gamma']
        self.rScaling = self.data['rScaling']
        self.lrFreq = self.data['lrFreq']

        # 3. load/save path configuration

        self.sPath = self.data['sPath']
        self.writeTMode = \
            True if 'writeTMode' not in keyList else self.data['writeTMode'] == "True"
        if self.writeTMode:
            time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
            self.tPath = self.data['tPath'] + self.data['envName']+time
            self.writer = SummaryWriter(self.tPath)
        self.lPath = self.data['lPath']

        # 4. miscelleous configuration
        self.device = self.data['device']
        self.inferMode = self.data['inferMode'] == "True"
        self.renderMode = self.data['renderMode'] == "True"
        self.keyList = keyList
        self.aSize = self.data['aSize']
        self.updateStep = self.data['updateStep']

        # 5. unity env setting
        id_ = np.random.randint(10, 100, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=2)
        setChannel = EnvironmentParametersChannel()
        imgMode = self.data['imgMode'] == "True"
        maxStack = self.data['maxStack']
        setChannel.set_float_parameter("maxStack", maxStack)
        self.imgMode = imgMode
        if imgMode:
            setChannel.set_float_parameter("imgMode", 1.0)
        else:
            setChannel.set_float_parameter("imgMode", 0)
        name = self.data['envName']
        self.nAgent = self.data['nAgent']
        setChannel.set_float_parameter("nAgent", self.nAgent)
        self.env = UnityEnvironment(
            name, worker_id=id_, 
            side_channels=[setChannel, engineChannel])
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs._dict.keys())[0]
        a = self.env.behavior_specs[self.behaviorNames]
        self.obsShape = a.observation_shapes[0][0]
        
        # replay memory for each agent
        self.replayMemory = deque(maxlen=30 * 100)

    def reset(self, id):
        self.replayMemory.clear()

    def genOptim(self):
        pass

    def initializePolicy(self):
        self.genOptim()

    def ppState(self, obs):
        pass

    def appendMemory(self, data):
        self.replayMemory.append(data)

    def getAction(self, state):
        pass

    def train(self):
        pass

    def run(self):
        pass

    def lrSheduling(self):
        pass
    
    def writeTrainInfo(self):
        self.info = """
        envName:{}
        lrFreq:{:3d}
        rScaling:{:3d}
        gamma:{}
        """.format(
            self.data['envName'], 
            self.lrFreq,
            self.rScaling, 
            self.gamma)
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
        networkKeyList = list(self.aData.keys())[:4]
        for key in networkKeyList:
            self.info += """
        network_{}:
        """.format(key)
            netList = list(self.aData[key].keys())
            for data in netList:
                self.info += """
                {}:{}
                """.format(data, self.aData[key][data])
 