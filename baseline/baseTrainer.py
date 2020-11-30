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

        # 2. set Hyper-parameter

        self.tau = self.data['tau']
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
        self.nAgent = self.data['env']['nAgent']

        # 5. Unity Setting
        id_ = np.random.randint(10, 100, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=4)
        setChannel = EnvironmentParametersChannel()
        envData = self.data['env']
        for key in envData.keys():
            setChannel.set_float_parameter(key, float(envData[key]))
        name = self.data['envName']
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