import torch
import numpy as np
# from PPO import Trainer


def CargoPPState(self, obs) -> tuple:
    vectorObs, imageObs = obs
    rState = torch.tensor(vectorObs[:, :8]).to(self.device).double()
    lidarPt = torch.tensor(vectorObs[:, 8:-1]).to(self.device).double()
    lidarPt = torch.unsqueeze(lidarPt, dim=1)
    imageObs = torch.tensor(imageObs).permute(0, 3, 1, 2).double().to(self.device)
    state = (rState, lidarPt, imageObs)
    return state


def CNN1DLTMPBatch(self, step, epoch, f):
    # self: PPO.Trainer.PPOOnPolicyTrainer
    k1 = self.data['K1']
    k2 = self.data['K2']
    div = int(k1/k2)
    rstate, lidarPt, image, action, reward, done = \
        [], [], [], [], [], []
    num_list = int(len(self.ReplayMemory_Trajectory)/k1)
    trstate, tlidarPt, tImage =\
        [[] for __ in range(num_list)],\
        [[] for __ in range(num_list)],\
        [[] for __ in range(num_list)]
    tState = [[] for _ in range(num_list - 1)]

    for data in self.replayMemory[0]:
        s, a, r, ns, d = data
        rstate.append(s[0])
        lidarPt.append(s[1])
        image.append(s[2])
        action.append(a)
        reward.append(r)
        done.append(d)
    z = 0
    for data in self.ReplayMemory_Trajectory:
        
        ts = data
        trstate[int(z/k1)].append(ts[0])
        tlidarPt[int(z/k1)].append(ts[1])
        tImage[int(z/k1)].append(ts[2])
        z += 1
    
    if len(trstate) == k1:
        zeroMode = True
    else:
        for _ in range(num_list - 1):
            tState[_] = (
                torch.cat(trstate[_], dim=0),
                torch.cat(tlidarPt[_], dim=0),
                torch.cat(tImage[_], dim=0))
        zeroMode = False
    rstate = torch.cat(rstate, dim=0)
    lidarPt = torch.cat(lidarPt, dim=0)
    image = torch.cat(image, dim=0)
    nrstate, nlidarPt, nimage = ns
    nrstate, nlidarPt, nimage =\
        torch.cat((rstate, nrstate), dim=0),\
        torch.cat((lidarPt, nlidarPt), dim=0),\
        torch.cat((image, nimage), dim=0)

    lidarPt = lidarPt.view((-1, self.nAgent, 1, self.sSize[-1]))
    rstate = rstate.view((-1, self.nAgent, 8))
    image = image.view((-1, self.nAgent, 1, 96, 96))
   
    nstate = (nrstate, nlidarPt, nimage)
    reward = np.array(reward)
    done = np.array(done)
    action = torch.tensor(action).to(self.device)

    self.agent.actor.zeroCellState()
    self.agent.critic.zeroCellState()
    self.copyAgent.actor.zeroCellState()
    self.copyAgent.critic.zeroCellState()

    if zeroMode is False:
        with torch.no_grad():
            for tr in tState:
                tr_cuda = tuple([x.to(self.device) for x in tr])
                self.agent.critic.forward(tr_cuda)
                self.copyAgent.critic.forward(tr_cuda)
                self.agent.actor.forward(tr_cuda)
                self.copyAgent.actor.forward(tr_cuda)
                del tr_cuda
            self.agent.actor.detachCellState()
            self.agent.critic.detachCellState()
            self.copyAgent.actor.detachCellState()
            self.copyAgent.critic.detachCellState()

    # 1. calculate the target value for actor and critic
    self.agent.actor.detachCellState()
    InitActorCellState = self.agent.actor.getCellState()
    InitCopyActorCellState = self.copyAgent.actor.getCellState()

    self.agent.critic.detachCellState()
    InitCriticCellState = self.agent.critic.getCellState()
    InitCopyCriticCellState = self.copyAgent.critic.getCellState()
    self.zeroGrad()
    # 2. implemented the training using the truncated BPTT
    for _ in range(epoch):
        self.agent.actor.setCellState(InitActorCellState)
        self.agent.critic.setCellState(InitCriticCellState)

        value = self.agent.critic.forward(nstate)[0]  # . step, nAgent, 1 -> -1, 1
        value = value.view(k1+1, self.nAgent, 1)
        nvalue = value[1:]
        value = value[:-1]
        gT, gAE = self.getReturn(reward, value, nvalue, done)
        gT = gT.view(k1, self.nAgent)
        gAE = gAE.view(k1, self.nAgent)

        self.agent.critic.setCellState(InitCriticCellState)
        self.copyAgent.actor.setCellState(InitCopyActorCellState)
        self.copyAgent.critic.setCellState(InitCopyCriticCellState)
        
        for i in range(div):
            _rstate = rstate[i*k2:(i+1)*k2].view(-1, 8)
            _lidarpt = lidarPt[i*k2:(1+i)*k2].view(-1, 1, self.sSize[-1])
            _image = image[i*k2:(1+i)*k2].view(-1, 1, 96, 96)
            _state = (_rstate, _lidarpt, _image)
            _action = action[i*k2:(i+1)*k2].view((-1, 2))
            _gT = gT[i*k2:(i+1)*k2].view(-1, 1)
            _gAE = gAE[i*k2:(i+1)*k2].view(-1, 1)
            _value = value[i*k2:(i+1)*k2].view(-1, 1)
            f(self, _state, _action, _gT, _gAE, _value, step, epoch)
            self.agent.actor.detachCellState()
            self.agent.critic.detachCellState()
        self.step(step+i, epoch)
        self.agent.actor.zeroCellState()
        self.agent.critic.zeroCellState()
        self.zeroGrad()
        if zeroMode is False:
            with torch.no_grad():
                for tr in tState:
                    tr_cuda = tuple([x.to(self.device) for x in tr])
                    self.agent.critic.forward(tr_cuda)
                    self.agent.actor.forward(tr_cuda)
                    del tr_cuda
                self.agent.critic.detachCellState()
                self.agent.actor.detachCellState()
        InitActorCellState = self.agent.actor.getCellState()
        InitCriticCellState = self.agent.critic.getCellState()
    
    del tState,  InitActorCellState, InitCriticCellState, \
        InitCopyActorCellState, InitCopyCriticCellState


def preprocessBatch(f):
    def wrapper(self, step, epoch):
        # MLPBatch(self, step, epoch, f)
        CNN1DLTMPBatch(self, step, epoch, f)
    return wrapper


def preprocessState(f):
    def wrapper(self, obs):
        # return self, MLPState(self, obs)
        # return CNN1DLTMPState(self, obs)
        return CargoPPState(self, obs)
    return wrapper