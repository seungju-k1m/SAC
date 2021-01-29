import ray
import time as tt
import datetime
import torch
import numpy as np
from PPO.Agent import ppoAgent
from PPO.wrapper import preprocessBatch
from baseline.utils import jsonParser, getOptim

from collections import deque

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter


path = './cfg/LSTMTrain.json'
parser = jsonParser(path)
data = parser.loadParser()
aData = parser.loadAgentParser()
optimData = parser.loadOptParser()
device = data['device']
writeMode = data['writeTMode']
tPath = data['tPath']
lPath = data['lPath']
sPath = data['sPath']


# define Constant

nEnv = data['nEnv']
nAgent = 64
TotalAgent = nEnv * nAgent
ReplayMemory = deque(maxlen=int(1e5))
ReplayMemory_Trajectory = deque(maxlen=int(1e5))
step = 0
ClipingNormCritic = 10
ClipingNormActor = 10


# Initialize Ray and Specify default data type of torch.tensor

ray.init(num_cpus=8)
torch.set_default_dtype(torch.float64)


# Load hyper-Parameter for Agent

entropyCoeff = data['entropyCoeff']
epsilon = data['epsilon']
lambda_ = data['lambda']
initLogStd = torch.tensor(data['initLogStd']).to(device)
finLogStd = torch.tensor(data['finLogStd']).to(device)
annealingStep = data['annealingStep']
LSTMNum = data['LSTMNum']
sSize = data['sSize']


gamma = data['gamma']
epoch = data['epoch']
updateOldP = data['updateOldP']


# Configure Writer

pureEnv = data['envName'].split('/')
name = pureEnv[-1]
time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
if writeMode:
    tPath = tPath + name + time
    writer = SummaryWriter(tPath)
sPath += name + '_' + str(time) + '.pth'


info = """
    Configuration for this experiment
    """


def writeDict(_data, key, n=0):
    global info
    tab = ""
    for _ in range(n):
        tab += '\t'
    if type(_data) == dict:
        for k in _data.keys():
            dK = _data[k]
            if type(dK) == dict:
                info += """
        {}{}:
            """.format(tab, k)
                writeDict(dK, k, n=n+1)
            else:
                info += """
        {}{}:{}
        """.format(tab, k, dK)
    else:
        info += """
        {}:{}
        """.format(key, _data)


def writeTrainInfo():
    global info
    key = data.keys()
    for k in key:
        _data = data[k]
        if type(_data) == dict:
            info += """
        {}:
        """.format(k)
            writeDict(_data, k, n=1)
        else:
            writeDict(_data, k)
    print(info)
    if writeMode:
        writer.add_text('Information', info, 0)


# In[10]:


writeTrainInfo()


# Instances for Agent

# In[11]:


Agent = ppoAgent(
    aData,
    coeff=entropyCoeff,
    epsilon=epsilon,
    device=device,
    initLogStd=initLogStd,
    finLogStd=finLogStd,
    annealingStep=annealingStep,
    LSTMNum=LSTMNum
)

if lPath != "None":
    Agent.load_state_dict(
        torch.load(lPath, map_location=device)
    )
    Agent.loadParameters()

OldAgent = ppoAgent(
    aData,
    coeff=entropyCoeff,
    epsilon=epsilon,
    device=device,
    initLogStd=initLogStd,
    finLogStd=finLogStd,
    annealingStep=annealingStep,
    LSTMNum=LSTMNum
)
OldAgent.update(Agent)

CopyAgent = ppoAgent(
    aData,
    coeff=entropyCoeff,
    epsilon=epsilon,
    device=device,
    initLogStd=initLogStd,
    finLogStd=finLogStd,
    annealingStep=annealingStep,
    LSTMNum=LSTMNum
)
CopyAgent.update(Agent)


# Configuration for Unity Environment

# In[12]:


_id = 32
time_scale = data['time_scale']
envData = data['env']
no_graphics = data['no_graphics']


engineChannel = EngineConfigurationChannel()
engineChannel.set_configuration_parameters(time_scale=time_scale)
setChannel = EnvironmentParametersChannel()
for key in envData.keys():
    setChannel.set_float_parameter(key, float(envData[key]))
name = data['envName']
envs = []
for i in range(nEnv):
    env = ray.remote(num_cpus=1)(UnityEnvironment)
    ENV = env.remote(
        name,
        worker_id=_id+i,
        side_channels=[setChannel, engineChannel],
        no_graphics=no_graphics,
        seed=1 + i * _id
    )
    ENV.reset.remote()
    envs.append(ENV)

behaviorNames = 'Robot?team=0'
for e in envs:
    ray.get(e._assert_behavior_exists.remote(behaviorNames))

print("""
Load the Unity Environment
""")


# Sampling and Training, AND Sampling,....


@ray.remote
def _getObs(env, behaviorNames, nAgent):
    done = [False for i in range(nAgent)]
    reward = [0 for i in range(nAgent)]
    decisionStep, terminalStep = ray.get(env.get_steps.remote(behaviorNames))
    obs, tobs = decisionStep.obs[0], terminalStep.obs[0]

    reward_, treward = decisionStep.reward, terminalStep.reward
    treward = np.array(treward)
    reward = reward_
    tAgentId = terminalStep.agent_id
    obsState = np.array(obs)
    k = 0
    for j, state in zip(tAgentId, tobs):
        obsState[j] = np.array(state)
        done[j] = True
        # reward[j] = treward[k]
        k += 1
    return (obsState, reward, treward, done)


def getObs(init=False) -> tuple:
    obsState = np.zeros((TotalAgent, 1447), dtype=np.float64)
    done = [False for i in range(TotalAgent)]
    reward = [0 for i in range(TotalAgent)]
    proc = []
    for i in range(nEnv):
        proc.append(_getObs.remote(
            envs[i],
            behaviorNames,
            nAgent
        ))
    for i in range(nEnv):
        t = ray.get(proc[i])
        s, r, r_, d = t
        obsState[i*nAgent:(i+1)*nAgent, :] = s
        done[i*nAgent:(i+1)*nAgent] = d
        if True in d:
            reward[i*nAgent:(i+1)*nAgent] = r_
        else:
            reward[i*nAgent:(i+1)*nAgent] = r
    if init:
        return obsState
    else:
        return (obsState, reward, done)


def ppState(obs) -> tuple:
    rState = torch.tensor(obs[:, :6]).to(device).double()
    lidarPt = torch.tensor(obs[:, 8:sSize[-1]+8]).to(device)
    lidarPt = torch.unsqueeze(lidarPt, dim=1).double()
    state = (rState, lidarPt)
    return state


def getAction(state) -> np.ndarray:
    with torch.no_grad():
        action = OldAgent.actorForward(state)
        action = action.cpu().numpy()
    return action


# In[16]:


def checkStep(action) -> None:
    for i in range(nEnv):
        act = ActionTuple(
            continuous=action[i*nAgent:(i+1)*nAgent, :]
        )
        envs[i].set_actions.remote(
            behaviorNames,
            act
        )
        envs[i].step.remote()


# In[17]:


Rewards = np.zeros(TotalAgent)
episodeReward = []


# In[18]:


def initSampling() -> tuple:
    init_obs = getObs(init=True)
    stateT = ppState(init_obs)
    action = getAction(stateT)

    return (stateT, action)


def Sampling(stateT, action) -> list:
    global Rewards
    global episodeReward
    global step
    with torch.no_grad():
        for i in range(160):
            checkStep(action)
            obs, reward, done = getObs()
            nstateT = ppState(obs)
            nAction = getAction(nstateT)
            ReplayMemory.append(
                (
                    stateT,
                    action.copy(),
                    reward,
                    nstateT,
                    done.copy()
                )
            )
            Rewards += reward

            stateT_cpu = tuple([x.cpu() for x in stateT])
            ReplayMemory_Trajectory.append(
                stateT_cpu
            )
            action = nAction
            stateT = nstateT
            step += 1
            Agent.decayingLogStd(step)
            OldAgent.decayingLogStd(step)
            CopyAgent.decayingLogStd(step)
            # print("InferenceTime:{:.3f}".format(tt.time() -z))
    
    return done


# Training
# 1. Generate Optimizer


def GenerateOptim() -> tuple:
    optimKeyList = list(optimData.keys())
    for key in optimKeyList:
        if key == "actor":
            aOptim = getOptim(
                optimData[key],
                Agent.actor.buildOptim())
        if key == "critic":
            cOptim = getOptim(
                optimData[key],
                Agent.critic.buildOptim()
            )
    return (aOptim, cOptim)


aOptim, cOptim = GenerateOptim()


# 2. Set Zero Gradient


def zeroGrad() -> None:
    aOptim.zero_grad()
    cOptim.zero_grad()


zeroGrad()


# 3. Train the Agent


def train(
    PPOAGENT,
    state,
    action,
    gT,
    gAE,
    critic,
    _step,
    _epoch
):
    PPOAGENT: ppoAgent
    lossC = PPOAGENT.calQLoss(
        state,
        gT.detach()
    )
    lossC.backward()
    
    minusObj, entropy = PPOAGENT.calAObj(
        CopyAgent,
        state,
        action,
        gT.detach() - critic.detach()
    )
    minusObj.backward()
    obj = minusObj.cpu().sum().detach().numpy()
    lossC = lossC.cpu().sum().detach().numpy()

    if writeMode:
        
        writer.add_scalar("Obj", -obj, _step+_epoch)
        writer.add_scalar("Critic Lostt", lossC, _step+_epoch)
        entropy = entropy.detach().cpu().numpy()
        writer.add_scalar("Entropy", entropy, _step + _epoch)

        gT = gT.view(-1)
        gT = torch.mean(gT).detach().cpu().numpy()
        writer.add_scalar("gT", gT, _step + _epoch)

        critic = critic.view(-1)
        critic = torch.mean(critic).detach().cpu().numpy()
        writer.add_scalar("critic", critic, _step + _epoch)


# In[24]:


def getReturn(
    reward,
    critic,
    nCritic,
    done
):
    gT, gAE = [], []
    length = len(reward)
    critic = critic.view(length, -1)
    nCritic = nCritic.view(length, -1)
    for i in range(TotalAgent):
        rA = reward[:, i]  # [step] , 100
        dA = done[:, i]  # [step] , 100
        cA = critic[:, i]
        ncA = nCritic[:, i] 
        GT = []
        GTDE = []
        discounted_Td = 0
        if dA[-1]:
            discounted_r = cA[-1]
        else:
            discounted_r = ncA[-1]

        for r, is_terminal, c, nc in zip(
                reversed(rA), 
                reversed(dA), 
                reversed(cA),
                reversed(ncA)):
            
            if is_terminal:
                td_error = r + gamma * c - c
            else:
                td_error = r + gamma * nc - c

            discounted_r = r + gamma * discounted_r
            discounted_Td = td_error + gamma * lambda_ * discounted_Td

            GT.append(discounted_r)
            GTDE.append(discounted_Td)
        GT = torch.tensor(GT[::-1]).view((-1, 1)).to(device)
        GTDE = torch.tensor(GTDE[::-1]).view((-1, 1)).to(device)
        gT.append(GT)
        gAE.append(GTDE)

    gT = torch.cat(gT, dim=0)
    gAE = torch.cat(gAE, dim=0)

    gT = gT.view(TotalAgent, -1)
    gT = gT.permute(1, 0).contiguous()
    gT = gT.view((-1, 1))

    gAE = gAE.view(TotalAgent, -1)
    gAE = gAE.permute(1, 0).contiguous()
    gAE = gAE.view((-1, 1)) 
    # seq, agent > seq * agent

    return gT, gAE


# In[25]:


def stepGradient(_step, _epoch):
    Agent.critic.clippingNorm(ClipingNormCritic)
    cOptim.step()
    Agent.actor.clippingNorm(ClipingNormActor)
    aOptim.step()

    normA = Agent.actor.calculateNorm().cpu().detach().numpy()
    normC = Agent.critic.calculateNorm().cpu().detach().numpy()
    if writeMode:
        
        writer.add_scalar('Action Gradient Mag', normA, _step+_epoch)
        writer.add_scalar('Critic Gradient Mag', normC, _step+_epoch)



def preprocessBatch(_step, _epoch):

    k1 = 160
    k2 = 10
    div = int(k1/k2)
    rstate, lidarPt, action, reward, done =         [], [], [], [], []
    num_list = int(len(ReplayMemory_Trajectory)/k1)
    trstate, tlidarPt = [[] for __ in range(num_list)], [[] for _ in range(num_list)]
    tState = [[] for _ in range(num_list - 1)]
    for ss in ReplayMemory:
        s, a, r, ns, d = ss
        rstate.append(s[0])
        lidarPt.append(s[1])
        action.append(a)
        reward.append(r)
        done.append(d)
    z = 0
    for ts in ReplayMemory_Trajectory:
        trstate[int(z/k1)].append(ts[0])
        tlidarPt[int(z/k1)].append(ts[1])
        z += 1
    if len(trstate) == k1:
        zeroMode = True
    else:
        for _ in range(num_list - 1):
            # print(trstate[_])
            tState[_] = (torch.cat(trstate[_], dim=0), torch.cat(tlidarPt[_], dim=0))
        zeroMode = False
    rstate = torch.cat(rstate, dim=0)
    lidarPt = torch.cat(lidarPt, dim=0)
    nrstate, nlidarPt = ns
    nrstate, nlidarPt = torch.cat((rstate, nrstate), dim=0), torch.cat((lidarPt, nlidarPt), dim=0)
    lidarPt = lidarPt.view((-1, TotalAgent, 1, sSize[-1]))
    rstate = rstate.view((-1, TotalAgent, 6))

    nstate = (nrstate, nlidarPt)

    reward = np.array(reward)
    done = np.array(done)
    action = torch.tensor(action).to(device)

    Agent.actor.zeroCellState()
    Agent.critic.zeroCellState()
    CopyAgent.actor.zeroCellState()
    CopyAgent.critic.zeroCellState()

    if zeroMode is False:
        with torch.no_grad():
            for tr in tState:
                tr_cuda = tuple([x.to(device) for x in tr])
                Agent.critic.forward(tr_cuda)
                Agent.actor.forward(tr_cuda)
                CopyAgent.critic.forward(tr_cuda)
                CopyAgent.actor.forward(tr_cuda)
                del tr_cuda
            Agent.actor.detachCellState()
            Agent.critic.detachCellState()
            CopyAgent.actor.detachCellState()
            CopyAgent.critic.detachCellState()
    
    InitActorCellState = Agent.actor.getCellState()
    InitCopyActorCellState = CopyAgent.actor.getCellState()

    InitCriticCellState = CopyAgent.actor.getCellState()
    InitCopyCriticCellState = CopyAgent.critic.getCellState()
    zeroGrad()

    for _ in range(epoch):
        Agent.actor.setCellState(InitActorCellState)
        Agent.critic.setCellState(InitCriticCellState)

        value = Agent.critic.forward(nstate)[0]
        value = value.view(k1+1, TotalAgent, 1)
        nvalue = value[1:]
        value = value[:-1]
        gT, gAE = getReturn(reward, value, nvalue, done)
        gT = gT.view(k1, TotalAgent)
        gAE = gAE.view(k1, TotalAgent)

        Agent.critic.setCellState(InitCriticCellState)
        CopyAgent.actor.setCellState(InitCopyActorCellState)
        CopyAgent.critic.setCellState(InitCopyCriticCellState)
        for i in range(div):
            _rstate = rstate[i*k2:(i+1)*k2].view(-1, 6)
            _lidarpt = lidarPt[i*k2:(1+i)*k2].view(-1, 1, sSize[-1])
            _state = (_rstate, _lidarpt)
            _action = action[i*k2:(i+1)*k2].view((-1, 2))
            _gT = gT[i*k2:(i+1)*k2].view(-1, 1)
            _gAE = gAE[i*k2:(i+1)*k2].view(-1, 1)
            _value = value[i*k2:(i+1)*k2].view(-1, 1)
            train(Agent, _state, _action, _gT, _gAE, _value, _step, _epoch)
            Agent.actor.detachCellState()
            Agent.critic.detachCellState()
        stepGradient(_step+i, _epoch)
        Agent.actor.zeroCellState()
        Agent.critic.zeroCellState()
        zeroGrad()
        if zeroMode is False:
            with torch.no_grad():
                for tr in tState:
                    tr_cuda = tuple([x.to(device) for x in tr])
                    Agent.critic.forward(tr_cuda)
                    Agent.actor.forward(tr_cuda)
                    del tr_cuda
                Agent.critic.detachCellState()
                Agent.actor.detachCellState()
        InitActorCellState = Agent.actor.getCellState()
        InitCriticCellState = Agent.critic.getCellState()
   

# initialize Sampling

# In[27]:


kz = 0
t = 0
while 1:
    
    stateT, action = initSampling()
    deltaTime = tt.time()
    done = Sampling(stateT, action)
    t += (tt.time() - deltaTime)
    preprocessBatch(step, epoch)
    kz += 1
    ReplayMemory.clear()
    if True in done:
        for e in envs:
            Agent.actor.zeroCellState()
            Agent.critic.zeroCellState()
            OldAgent.actor.zeroCellState()
            OldAgent.critic.zeroCellState()
            CopyAgent.actor.zeroCellState()
            CopyAgent.critic.zeroCellState()
            ReplayMemory_Trajectory.clear()
            e.step.remote()
    if kz == updateOldP:
        OldAgent.update(Agent)
        CopyAgent.update(Agent)
        kz = 0

    if step % 4000 == 0:
        _reward = Rewards.mean()
        if writeMode:
            writer.add_scalar('Performance', _reward, step)
        print("""
        Step : {:5d} // Performance : {:.3f} // Inference:{:.3f}
        """.format(step, _reward, t/4000))
        t = 0
        Rewards = np.zeros(TotalAgent)
        torch.save(Agent.state_dict(), sPath)
