#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch.utils.tensorboard import SummaryWriter


# In[2]:


path = './cfg/WOImage.json'
parser = jsonParser(path)
data = parser.loadParser()
aData = parser.loadAgentParser()
optimData = parser.loadOptParser()
device = data['device']
writeMode = data['writeTMode']
tPath = data['tPath']
lPath = data['lPath']
sPath = data['sPath']
k1 = data['K1']
k2 = data['K2']


# define Constant

# In[3]:


nEnv = data['nEnv']
nAgent = 8
TotalAgent = nEnv * nAgent
ReplayMemory = deque(maxlen=int(1e5))
ReplayMemory_Trajectory = deque(maxlen=int(1e5))
step = 0
ClipingNormCritic = 100000
ClipingNormActor = 100000


# Initialize Ray and Specify default data type of torch.tensor

# In[4]:


ray.init(num_cpus=8)
torch.set_default_dtype(torch.float64)


# Load hyper-Parameter for Agent

# In[5]:


entropyCoeff = data['entropyCoeff']
epsilon = data['epsilon']
lambda_ = data['lambda']
initLogStd = torch.tensor(data['initLogStd']).to(device)
finLogStd = torch.tensor(data['finLogStd']).to(device)
annealingStep = data['annealingStep']
LSTMName = data['LSTMName']
sSize = data['sSize']


# In[6]:


gamma = data['gamma']
epoch = data['epoch']
updateOldP = data['updateOldP']


# Configure Writer

# In[7]:


pureEnv = data['envName'].split('/')
name = pureEnv[-1]
time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
if writeMode:
    tPath = tPath + name + time
    writer = SummaryWriter(tPath)
sPath += name + '_' + str(time) +'.pth'


# In[8]:


info =     """
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
                info +=            """
        {}{}:
            """.format(tab, k)
                writeDict(dK, k, n=n+1)
            else:
                info +=         """
        {}{}:{}
        """.format(tab, k, dK)
    else:
        info +=        """
        {}:{}
        """.format(key, _data)

def writeTrainInfo():
    global info
    key = data.keys()
    for k in key:
        _data = data[k]
        if type(_data) == dict:
            info +=        """
        {}:
        """.format(k)
            writeDict(_data, k ,n=1)
        else:
            writeDict(_data, k)
    print(info)
    if writeMode:
        writer.add_text('Information', info, 0)


# In[9]:


writeTrainInfo()


# Instances for Agent

# In[10]:


Agent = ppoAgent(
    aData,
    coeff=entropyCoeff,
    epsilon=epsilon,
    device=device,
    initLogStd=initLogStd,
    finLogStd=finLogStd,
    annealingStep=annealingStep,
    LSTMName=LSTMName
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
    LSTMName=LSTMName
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
    LSTMName=LSTMName
)
CopyAgent.update(Agent)


# Configuration for Unity Environment

# In[11]:


_id = 320
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
        seed = 1 + i * _id
    )
    ENV.reset.remote()
    envs.append(ENV)

behaviorNames = 'Agent?team=0'
for e in envs:
    ray.get(e._assert_behavior_exists.remote(behaviorNames))

print("""
Load the Unity Environment
""")


# Sampling and Training, AND Sampling,....

# In[12]:


@ray.remote
def _getObs(env, behaviorNames, nAgent, init):
    decisionStep, terminalStep = ray.get(env.get_steps.remote(behaviorNames))
    image = decisionStep.obs[0]
    obs = decisionStep.obs[1]
    rewards = decisionStep.reward
    obs = obs.tolist()

    obs = list(map(lambda x: np.array(x), obs))
    obs = np.array(obs)

    done = []
    
    for done_idx in obs[:, -1]:
        done.append(done_idx == 1)
    reward = rewards
    
    obsState = (obs, image)

    if init:
        return obsState
    else:
        return(obsState, reward, done)

def getObs(init=False) -> tuple:
    done = [False for i in range(TotalAgent)]
    reward = [0 for i in range(TotalAgent)]
    proc = []
    vectorObs, imageObs = np.zeros((TotalAgent, 369)), np.zeros((TotalAgent, 96, 96, 1))
    for i in range(nEnv):
        proc.append(_getObs.remote(
            envs[i],
            behaviorNames,
            nAgent,
            init
        ))
    for i in range(nEnv):
        t = ray.get(proc[i])
        if init:
            s = t
            vectorObs[i*nAgent:(i+1)*nAgent] = s[0]
            imageObs[i*nAgent:(i+1)*nAgent] = s[1]
        else:
            s, r, d = t
            vectorObs[i*nAgent:(i+1)*nAgent] = s[0]
            imageObs[i*nAgent:(i+1)*nAgent] = s[1]
            done[i*nAgent:(i+1)*nAgent] = d
            reward[i*nAgent:(i+1)*nAgent] = r
    
    obsState = (vectorObs, imageObs)
    if init:
        return obsState
    else:
        return (obsState, reward, done)


# In[13]:


def ppState(obs) -> tuple:
    vectorObs, imageObs = obs
    rState = torch.tensor(vectorObs[:, :8]).to(device).double()
    lidarPt = torch.tensor(vectorObs[:, 8:-1]).to(device).double()
    lidarPt = torch.unsqueeze(lidarPt, dim=1)
    state = (rState, lidarPt)
    return state


# In[14]:


def getAction(state) -> np.ndarray:
    with torch.no_grad():
        action = OldAgent.actorForward(state)
        action = action.cpu().numpy()
    return action


# In[15]:


def checkStep(action) -> None:
    for i in range(nEnv):
        act = action[i*nAgent:(i+1)*nAgent]
        envs[i].set_actions.remote(
            behaviorNames,
            act
        )
        envs[i].step.remote()


# In[16]:


Rewards = np.zeros(TotalAgent)
episodeReward = []


# Training

# 1. Generate Optimizer

# In[17]:


def GenerateOptim() -> tuple:
    optimKeyList = list(optimData.keys())
    for key in optimKeyList:
        if key == "actor":
            aOptim = getOptim(
                optimData[key],
                Agent.actor.buildOptim())
    return aOptim


# In[18]:


aOptim = GenerateOptim()


# 2. Set Zero Gradient

# In[19]:


def zeroGrad() -> None:
    aOptim.zero_grad()


# In[20]:


zeroGrad()


# 3. Train the Agent

# In[21]:


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
    PPOAGENT:ppoAgent
    lossC, minusObj, entropy = PPOAGENT.calLoss(
            CopyAgent,
            state,
            action.detach(),
            gT.detach(),
            critic.detach(),
            gAE.detach()
        )

    objectFunction = minusObj + lossC
    objectFunction.backward()
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

        gAE = gAE.view(-1)
        gAE = torch.mean(gAE).detach().cpu().numpy()
        writer.add_scalar("gAE", gAE, _step + _epoch)


# In[22]:


def getReturn(
    reward,
    critic,
    nCritic,
    done
)->tuple:
    gT, gAE = [], []
    step = len(reward)
    critic = critic.view((step, -1))
    nCritic = nCritic.view((step, -1))
    for i in range(TotalAgent):
        rA = reward[:, i]  # 160
        dA = done[:, i]  # 160
        cA = critic[:, i] 
        ncA = nCritic[:, i] 
        GT = []
        GTDE = []
        discounted_Td = 0
        discounted_r = ncA[-1]

        for r, is_terminal, c, nc in zip(
                reversed(rA), 
                reversed(dA), 
                reversed(cA),
                reversed(ncA)):
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

    gT = gT.view(nAgent, -1)
    gT = gT.permute(1, 0).contiguous()
    gT = gT.view((-1, 1))

    gAE = gAE.view(nAgent, -1)
    gAE = gAE.permute(1, 0).contiguous()
    gAE = gAE.view((-1, 1))

    return gT, gAE


# In[23]:


def stepGradient(_step, _epoch):
    Agent.actor.clippingNorm(ClipingNormActor)
    aOptim.step()

    normA = Agent.actor.calculateNorm().cpu().detach().numpy()

    if writeMode:
        writer.add_scalar('Action Gradient Mag', normA, _step+_epoch)


# In[24]:


def preprocessBatch(_step, _epoch):
    div = int(k1/k2)

    # Ready for the batch-preprocessing
    rstate, lidarPt, action, reward, done =         [], [], [], [], []
    num_list = int(len(ReplayMemory_Trajectory)/k1)
    trstate, tlidarPt =        [[] for __ in range(num_list)],        [[] for __ in range(num_list)]
    tState = [[] for _ in range(num_list - 1)]

    # get the samples from the replayMemory
    for data in ReplayMemory:
        s, a, r, ns, d = data
        rstate.append(s[0])
        lidarPt.append(s[1])
        action.append(a)
        reward.append(r)
        done.append(d)
    
    # z can be thought as the number for slicing the trajectory value
    # by slicing the trajectory samples, reduce the memory usage.
    z = 0
    for data in ReplayMemory_Trajectory:
        
        ts = data
        trstate[int(z/k1)].append(ts[0])
        tlidarPt[int(z/k1)].append(ts[1])
        z += 1
    
    # First K1 Horizon, there is no need to prepare the trajectory.
    if len(trstate) == k1:
        zeroMode = True
    else:
        for _ in range(num_list - 1):
            tState[_] = (
                torch.cat(trstate[_], dim=0),
                torch.cat(tlidarPt[_], dim=0))
        zeroMode = False
    
    # Second preprocess-batch
    rstate = torch.cat(rstate, dim=0)
    lidarPt = torch.cat(lidarPt, dim=0)
    nrstate, nlidarPt = ns

    # nrstate, nlidarPt have K1+1 elements
    nrstate, nlidarPt =        torch.cat((rstate, nrstate), dim=0),        torch.cat((lidarPt, nlidarPt), dim=0)
    nstate = (nrstate, nlidarPt)

    # viewing the tensor, sequence, nAgent, data
    # this form for BPTT.
    lidarPt = lidarPt.view((-1, TotalAgent, 1, 360))
    rstate = rstate.view((-1, TotalAgent, 8))
   
    # data casting.
    reward = np.array(reward)
    done = np.array(done)
    action = torch.tensor(action).to(device)

    # initalize the cell state of agent at the 0 step.
    Agent.actor.zeroCellState()
    CopyAgent.actor.zeroCellState()

    # 0. get the cell state before the K1 Step.
    # To do this, we use trajectory samples by just forwarding them.
    if zeroMode is False:
        with torch.no_grad():
            for tr in tState:
                tr_cuda = tuple([x.to(device) for x in tr])
                Agent.actor.forward(tr_cuda)
                CopyAgent.actor.forward(tr_cuda)
                del tr_cuda
            # detaching!!
            Agent.actor.detachCellState()
            CopyAgent.actor.detachCellState()

    Agent.actor.detachCellState()
    InitActorCellState = Agent.actor.getCellState()
    InitCopyActorCellState = CopyAgent.actor.getCellState()
    zeroGrad()

    # 2. implemented the training using the truncated BPTT
    for _ in range(epoch):
        # reset the agent at previous K1 step.
        Agent.actor.setCellState(InitActorCellState)

        # by this command, cell state of agent reaches the current Step.
        value = Agent.criticForward(nstate)
        
        # calculate the target value for training
        value = value.view(k1+1, TotalAgent, 1)
        nvalue = value[1:]
        value = value[:-1]
        gT, gAE = getReturn(reward, value, nvalue, done)
        gT = gT.view(k1, TotalAgent)
        gAE = gAE.view(k1, TotalAgent)

        # before training, reset the cell state of agent at Previous K1 step.
        CopyAgent.actor.setCellState(InitCopyActorCellState)
        Agent.actor.setCellState(InitActorCellState)
        
        # div can be thought as slice size for BPTT.
        for i in range(div):
            # ready for the batching.
            _rstate = rstate[i*k2:(i+1)*k2].view(-1, 8).detach()
            _lidarpt = lidarPt[i*k2:(1+i)*k2].view(-1, 1, sSize[-1]).detach()
            _state = (_rstate, _lidarpt)
            _action = action[i*k2:(i+1)*k2].view((-1, 2))
            _gT = gT[i*k2:(i+1)*k2].view(-1, 1)
            _gAE = gAE[i*k2:(i+1)*k2].view(-1, 1)
            _value = value[i*k2:(i+1)*k2].view(-1, 1)

            # after calling f, cell state would jump K2 Step from the previous Step.
            train(Agent, _state, _action, _gT, _gAE, _value, step, epoch)

            # detaching device for BPTT
            Agent.actor.detachCellState()
        
        # step the gradient for updating
        stepGradient(step+i, epoch)
        zeroGrad()

        # get the new cell state of new agent
        # Initialize the agent at 0 step.
        Agent.actor.zeroCellState()
        if zeroMode is False:
            with torch.no_grad():
                for tr in tState:
                    tr_cuda = tuple([x.to(device) for x in tr])
                    Agent.actor.forward(tr_cuda)
                    del tr_cuda
                Agent.actor.detachCellState()
        InitActorCellState = Agent.actor.getCellState()
    
    del tState,  InitActorCellState,          InitCopyActorCellState
   


# initialize Sampling

# In[25]:


step = 0
episodeReward = []
k = 0
Rewards = np.zeros(TotalAgent)
obs = getObs(init=True)
stateT = ppState(obs)
action = getAction(stateT)
while 1:
    checkStep(action)

    obs, reward, done = getObs()

    Rewards += reward

    nStateT = ppState(obs)
    nAction = getAction(nStateT)

    with torch.no_grad():
        ReplayMemory.append(
            (
                stateT,
                action.copy(),
                reward.copy(),
                nStateT,
                done
            )
        )
        stateT_cpu = tuple([x.cpu() for x in stateT])
        ReplayMemory_Trajectory.append(
            stateT_cpu
        )
    
    action = nAction
    stateT = nStateT
    step += 1

    Agent.decayingLogStd(step)
    CopyAgent.decayingLogStd(step)
    OldAgent.decayingLogStd(step)

    if (step) % (data['K1']) == 0:
        k += 1
        preprocessBatch(step, epoch)
        ReplayMemory.clear()

        if k % updateOldP == 0:
            OldAgent.update(Agent)
            CopyAgent.update(Agent)
            k = 0
    
    if True in done:
        Agent.actor.zeroCellState()
        OldAgent.actor.zeroCellState()
        CopyAgent.actor.zeroCellState()
        ReplayMemory_Trajectory.clear()
        for env in envs:
            env.step.remote()
        
        obs = getObs(init=True)
        stateT = ppState(obs)
        action = getAction(stateT)
    
    if step % 1000 == 0:
        episodeReward = np.array(Rewards)
        reward = episodeReward.mean()

        if writeMode:
            writer.add_scalar("Reward", reward, step)
        print("""
                Step : {:5d} // Reward : {:.3f}  
                """.format(step, reward))
        
        Rewards = np.zeros(TotalAgent)
        torch.save(Agent.state_dict(), sPath)
