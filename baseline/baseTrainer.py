import torch
import datetime
import numpy as np

from baseline.utils import jsonParser
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torch.utils.tensorboard import SummaryWriter


class OFFPolicy:

    def __init__(self, fName):
        # 1. load parser
        parser = jsonParser(fName)
        self.data = parser.loadParser()
        self.aData = parser.loadAgentParser()
        self.optimData = parser.loadOptParser()

        keyList = list(self.data.keys())
        torch.manual_seed(self.data['seed'])

        # 2. set Hyper-parameter

        self.tau = self.data['tau']
        
        name = self.data['envName']
        
        self.nReplayMemory = int(self.data['nReplayMemory'])
        self.gamma = self.data['gamma']
        self.bSize = self.data['bSize']
        self.rScaling = self.data['rScaling']
        self.lrFreq = self.data['lrFreq']
        self.startStep = self.data['startStep']

        # 3. load/save path configuration

        self.sPath = self.data['sPath']
        self.writeTMode = self.data['writeTMode']
        if self.writeTMode:
            time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
            self.tPath = self.data['tPath'] + self.data['envName']+time
            self.writer = SummaryWriter(self.tPath)
        self.lPath = self.data['lPath']

        # 4. miscelleous configuration
        self.device = self.data['device']
        self.inferMode = self.data['inferMode']
        self.renderMode = self.data['renderMode']
        self.keyList = keyList
        self.aSize = self.data['aSize']
        self.sSize = self.data['sSize']
        self.nAgent = self.data['nAgent']

        # 5. Unity Setting
        id_ = np.random.randint(10, 100, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=2)
        setChannel = EnvironmentParametersChannel()
        resolution = self.data['resolution']
        imgMode = self.data['imgMode']
        maxStack = self.data['maxStack']
        coeffMAngV = self.data['coeffMAngV']
        coeffAngV = self.data['coeffAngV']
        coeffDDist = self.data['coeffDDist']
        coeffInnerProduct = self.data['coeffInnerProduct']
        coeffIntegral = self.data['coeffIntegral']
        endReward = self.data['endReward']
        objRewardN = self.data['objRewardN']
        
        setChannel.set_float_parameter("coeffIntegral", coeffIntegral)
        setChannel.set_float_parameter("maxStack", maxStack)
        setChannel.set_float_parameter('resolution', resolution)
        setChannel.set_float_parameter('coeffMAngV', coeffMAngV)
        setChannel.set_float_parameter('coeffAngV', coeffAngV)
        setChannel.set_float_parameter('coeffDDist', coeffDDist)
        setChannel.set_float_parameter('endReward', endReward)
        setChannel.set_float_parameter('objRewardN', objRewardN)
        setChannel.set_float_parameter('coeffInnerProduct', coeffInnerProduct)
        setChannel.set_float_parameter('resolution', resolution)
        
        self.imgMode = imgMode
        if imgMode:
            setChannel.set_float_parameter("imgMode", 1.0)
        else:
            setChannel.set_float_parameter("imgMode", 0)
        name = self.data['envName']
        self.nAgent = self.data['nAgent']
        Count = self.data['Count']
        setChannel.set_float_parameter("nAgent", self.nAgent)
        setChannel.set_float_parameter("Count", Count)
        self.env = UnityEnvironment(
            name, worker_id=id_, 
            side_channels=[setChannel, engineChannel])
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs._dict.keys())[0]

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
        self.info = "Configuration of this Experiment"
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
        for key in networkKeyList[:-1]:
            self.info += """
        network_{}:
        """.format(key)
            netList = list(self.aData[key].keys())
            for data in netList:
                self.info += """
                {}:{}
                """.format(data, self.aData[key][data])