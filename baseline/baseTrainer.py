import ray
import datetime
import numpy as np

from baseline.utils import jsonParser
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torch.utils.tensorboard import SummaryWriter


"""
Reinforcement Learning내의 알고리즘은 크게 OFFPolicy, OnPolicy로 나눌 수 있다.
알고리즘들은 공통점을 가지고 있기 때문에, 
코드의 생산성을 위해서 공통된 생성자 선언과 method선언을 baseTrainer module에서 해준다.
"""


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
        self.nAgent = self.data['env']['Number_Agent']

    def LoadUnityEnv(self):
        # 5. Unity Setting
        id_ = np.random.randint(10, 100, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=self.data['time_scale'])
        setChannel = EnvironmentParametersChannel()
        envData = self.data['env']
        for key in envData.keys():
            setChannel.set_float_parameter(key, float(envData[key]))
        name = self.data['envName']
        self.envs = []
        for i in range(2):
            self.envs.append(UnityEnvironment(
                name, worker_id=id_ + i,
                side_channels=[setChannel, engineChannel]))
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs._dict.keys())[0]

    def LoadGymEnv(self):
        pass

    def reset(self):
        """
        deprecated.
        """
        pass

    def genOptim(self):
        """
        optimizer를 생성하는 method이다.
        """
        pass

    def initializePolicy(self):
        """
        deprecated
        """
        self.reset()
        self.genOptim()

    def ppState(self, obs):
        """
        observation을 전처리하여 state로 변환한다.
        """
        pass

    def appendMemory(self, data):
        pass

    def getAction(self, state):
        """
        state를 입력으로 받고 action을 출력한다.
        """
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

    def writeDict(self, data, key, n=0):
        tab = ""
        for _ in range(n):
            tab += '\t'
        if type(data) == dict:
            for k in data.keys():
                dK = data[k]
                if type(dK) == dict:
                    self.info +=\
                """
            {}{}:
                """.format(tab, k)
                    self.writeDict(dK, k, n=n+1)
                else:
                    self.info += \
            """
            {}{}:{}
            """.format(tab, k, dK)
        else:
            self.info +=\
            """
            {}:{}
            """.format(key, data)
    
    def writeTrainInfo(self):
        self.info = """
        Configuration for this experiment
        """
        key = self.data.keys()
        for k in key:
            data = self.data[k]
            if type(data) == dict:
                self.info +=\
            """
            {}:
            """.format(k)
                self.writeDict(data, k, n=1)
            else:
                self.writeDict(data, k)

        print(self.info)
        self.writer.add_text('info', self.info, 0)


def getSpec(self):
    return self._env_specs


class ONPolicy:
    
    def __init__(self, fName):
        # 1. load parser
        parser = jsonParser(fName)
        self.data = parser.loadParser()
        self.aData = parser.loadAgentParser()
        self.optimData = parser.loadOptParser()

        keyList = list(self.data.keys())

        # 2. set Hyper-parameter

        self.gamma = self.data['gamma']
        self.rScaling = self.data['rScaling']
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
        self.nAgent = self.data['env']['Number_Agent']

    def loadUnityEnv(self):
        # Unity Setting
        id_ = np.random.randint(10, 100, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=self.data['time_scale'])
        setChannel = EnvironmentParametersChannel()
        envData = self.data['env']
        for key in envData.keys():
            setChannel.set_float_parameter(key, float(envData[key]))
        name = self.data['envName']
        self.envs = []
        self.nEnv = self.data['nEnv']
        nEnv = self.nEnv
        # setattr(UnityEnvironment, "")
        for i in range(nEnv):
            env = ray.remote(UnityEnvironment)
            env.options(num_cpus=8)
            actor = env.remote(
                name,
                worker_id=id_+i,
                side_channels=[setChannel, engineChannel],
                no_graphics=self.data['no_graphics'])
            actor.reset.remote()
            self.envs.append(actor)
        # for e in self.envs:
        #     e.reset.remote()
        self.behaviorNames = 'Robot?team=0'
        for e in self.envs:
            ray.get(e._assert_behavior_exists.remote(self.behaviorNames))
        print("Hello")

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

    def writeDict(self, data, key, n=0):
        tab = ""
        for _ in range(n):
            tab += '\t'
        if type(data) == dict:
            for k in data.keys():
                dK = data[k]
                if type(dK) == dict:
                    self.info +=\
                """
            {}{}:
                """.format(tab, k)
                    self.writeDict(dK, k, n=n+1)
                else:
                    self.info += \
            """
            {}{}:{}
            """.format(tab, k, dK)
        else:
            self.info +=\
            """
            {}:{}
            """.format(key, data)
    
    def writeTrainInfo(self):
        self.info = """
        Configuration for this experiment
        """
        key = self.data.keys()
        for k in key:
            data = self.data[k]
            if type(data) == dict:
                self.info +=\
            """
            {}:
            """.format(k)
                self.writeDict(data, k, n=1)
            else:
                self.writeDict(data, k)

        print(self.info)
        if self.writeTMode:
            self.writer.add_text('info', self.info, 0)  