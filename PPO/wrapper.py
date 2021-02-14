import os
import gc
import sys
import copy
import psutil
import torch
import numpy as np
# from PPO.Trainer import PPOOnPolicyTrainer


def CargoPPState(self, obs) -> tuple:
    vectorObs, imageObs = obs
    rState = torch.tensor(vectorObs[:, :8]).to(self.device).double()
    lidarPt = torch.tensor(vectorObs[:, 8:-1]).to(self.device).double()
    lidarPt = torch.unsqueeze(lidarPt, dim=1)
    imageObs = torch.tensor(imageObs).permute(0, 3, 1, 2).double().to(self.device)
    state = (rState, lidarPt, imageObs)
    del obs
    return state


def CargoWOIPPState(self, obs) -> tuple:
    vectorObs, imageObs = obs
    # rState = torch.tensor(vectorObs[:, :8]).double().detach()
    # lidarPt = torch.tensor(vectorObs[:, 8:-1]).double().detach()
    # lidarPt = torch.unsqueeze(lidarPt, dim=1)
    rState = vectorObs[:, :8]
    lidarPt = vectorObs[:, 8:-1]
    lidarPt = np.expand_dims(lidarPt, axis=1)
    state = (rState, lidarPt)
    return state


def CargoWOIBatch(self, step, epoch, f):
    # self: PPOOnPolicyTrainer

    # Specify the info of Horizon
    # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
    k1 = 160
    k2 = 10
    div = int(k1/k2)

    # Ready for the batch-preprocessing
    rstate, lidarPt, action, reward, done = \
        [], [], [], [], []
    num_list = int(len(self.ReplayMemory_Trajectory)/k1)
    trstate, tlidarPt =\
        [[] for __ in range(num_list)],\
        [[] for __ in range(num_list)]
    tState = [[] for _ in range(num_list - 1)]

    # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))

    # get the samples from the replayMemory
    for data in self.replayMemory[0]:
        s, a, r, ns, d = data
        rstate.append(torch.from_numpy(copy.deepcopy(s[0])).to(self.device).double())
        lidarPt.append(torch.from_numpy(copy.deepcopy(s[1])).to(self.device).double())
        action.append(copy.deepcopy(a))
        reward.append(copy.deepcopy(r))
        done.append(copy.deepcopy(d))
        
    # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
    
    # z can be thought as the number for slicing the trajectory value
    # by slicing the trajectory samples, reduce the memory usage.
    z = 0
    for data in self.ReplayMemory_Trajectory:
        trstate[int(z/k1)].append(torch.from_numpy(copy.deepcopy(data[0])).double())
        tlidarPt[int(z/k1)].append(torch.from_numpy(copy.deepcopy(data[1])).double())
        z += 1
    # First K1 Horizon, there is no need to prepare the trajectory.
    if z == k1:
        zeroMode = True
    else:
        for _ in range(num_list - 1):
            tState[_] = (
                torch.cat(trstate[_], dim=0),
                torch.cat(tlidarPt[_], dim=0))
        zeroMode = False
    
    # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
    
    # Second preprocess-batch
    rstate = torch.cat(rstate, dim=0)
    lidarPt = torch.cat(lidarPt, dim=0)
    _nrstate, _nlidarPt = ns

    _nrstate, _nlidarPt = \
        torch.from_numpy(_nrstate).to(self.device).double(),  \
        torch.from_numpy(_nlidarPt).to(self.device).double()

    # nrstate, nlidarPt have K1+1 elements
    nrstate, nlidarPt =\
        torch.cat((rstate, _nrstate), dim=0),\
        torch.cat((lidarPt, _nlidarPt), dim=0)
    nstate = (nrstate, nlidarPt)

    # viewing the tensor, sequence, nAgent, data
    # this form for BPTT.
    lidarPt = lidarPt.view((-1, self.nAgent, 1, self.sSize[-1]))
    rstate = rstate.view((-1, self.nAgent, 8))

    # data casting.
    reward = np.array(reward)
    done = np.array(done)
    action = torch.tensor(action).to(self.device)

    # initalize the cell state of agent at the 0 step.
    self.agent.actor.zeroCellState()
    self.copyAgent.actor.zeroCellState()

    # 0. get the cell state before the K1 Step.
    # To do this, we use trajectory samples by just forwarding them.
    if zeroMode is False:
        for tr in self.tState:
            # tr_cuda = tuple([x.to(self.device) for x in tr])
            self.agent.actor.forward(tuple([x.to(self.device) for x in tr]))
            self.copyAgent.actor.forward(tuple([x.to(self.device) for x in tr]))
        # detaching!!
        self.agent.actor.detachCellState()
        self.copyAgent.actor.detachCellState()

    self.agent.actor.detachCellState()
    InitActorCellState = self.agent.actor.getCellState()
    InitCopyActorCellState = self.copyAgent.actor.getCellState()
    self.zeroGrad()

    # 2. implemented the training using the truncated BPTT
    for _ in range(epoch):
        # reset the agent at previous K1 step.
        # by this command, cell state of agent reaches the current Step.
        with torch.no_grad():
            self.agent.actor.setCellState(InitActorCellState)
            value = self.agent.criticForward(nstate)
            
            # calculate the target value for training
            value = value.view(k1+1, self.nAgent, 1)
            nvalue = value[1:]
            value = value[:-1]
            gT, gAE = self.getReturn(reward, value, nvalue, done)
            gT = gT.view(k1, self.nAgent)
            gAE = gAE.view(k1, self.nAgent)

            # before training, reset the cell state of agent at Previous K1 step.
            self.copyAgent.actor.setCellState(InitCopyActorCellState)
            self.agent.actor.setCellState(InitActorCellState)
        # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
        
        # div can be thought as slice size for BPTT.
        for i in range(div):
            # ready for the batching.
            _rstate = rstate[i*k2:(i+1)*k2].view(-1, 8).detach()
            _lidarpt = lidarPt[i*k2:(1+i)*k2].view(-1, 1, self.sSize[-1]).detach()
            _state = (_rstate, _lidarpt)
            _action = action[i*k2:(i+1)*k2].view((-1, 2))
            _gT = gT[i*k2:(i+1)*k2].view(-1, 1)
            _gAE = gAE[i*k2:(i+1)*k2].view(-1, 1)
            _value = value[i*k2:(i+1)*k2].view(-1, 1)

            # after calling f, cell state would jump K2 Step from the previous Step.
            f(self, _state, _action, _gT, _gAE, _value, step, epoch)

            # detaching device for BPTT
            self.agent.actor.detachCellState()
            # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
        
        # step the gradient for updating
        self.step(step+i, epoch)
        self.zeroGrad()
        # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
        # get the new cell state of new agent
        # Initialize the agent at 0 step.
        self.agent.actor.zeroCellState()
        if zeroMode is False:
            with torch.no_grad():
                for tr in self.tState:
                    # tr_cuda = tuple([x.to(self.device) for x in tr])
                    self.agent.actor.forward(tuple([x.to(self.device) for x in tr]))
                self.agent.actor.detachCellState()
        InitActorCellState = self.agent.actor.getCellState()
        
        # print("Current Memory : {:.3f}".format(current_process.memory_info()[0]/2.**20))
    self.agent.actor.detachCellState()
    self.copyAgent.actor.detachCellState()

    self.replayMemory[0].clear()


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
    self.copyAgent.actor.zeroCellState()

    if zeroMode is False:
        with torch.no_grad():
            for tr in tState:
                tr_cuda = tuple([x.to(self.device) for x in tr])
                self.agent.actor.forward(tr_cuda)
                self.copyAgent.actor.forward(tr_cuda)
                del tr
                del tr_cuda
            self.agent.actor.detachCellState()
            self.copyAgent.actor.detachCellState()

    # 1. calculate the target value for actor and critic
    self.agent.actor.detachCellState()
    InitActorCellState = self.agent.actor.getCellState()
    InitCopyActorCellState = self.copyAgent.actor.getCellState()
    self.zeroGrad()

    # 2. implemented the training using the truncated BPTT
    for _ in range(epoch):
        self.agent.actor.setCellState(InitActorCellState)

        value = self.agent.criticForward(nstate)[0]  # . step, nAgent, 1 -> -1, 1
        value = value.view(k1+1, self.nAgent, 1)
        nvalue = value[1:]
        value = value[:-1]
        gT, gAE = self.getReturn(reward, value, nvalue, done)
        gT = gT.view(k1, self.nAgent)
        gAE = gAE.view(k1, self.nAgent)

        self.copyAgent.actor.setCellState(InitCopyActorCellState)
        
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
        self.step(step+i, epoch)
        self.agent.actor.zeroCellState()
        self.zeroGrad()
        if zeroMode is False:
            with torch.no_grad():
                for tr in tState:
                    tr_cuda = tuple([x.to(self.device) for x in tr])
                    self.agent.actor.forward(tr_cuda)
                    del tr
                    del tr_cuda
                self.agent.actor.detachCellState()
        InitActorCellState = self.agent.actor.getCellState()
    
    del tState,  InitActorCellState,  \
        InitCopyActorCellState


def preprocessBatch(f):
    def wrapper(self, step, epoch):
        # CNN1DLTMPBatch(self, step, epoch, f)
        CargoWOIBatch(self, step, epoch, f)
    return wrapper


def preprocessState(f):
    def wrapper(self, obs):
        # return CargoPPState(self, obs)
        return CargoWOIPPState(self, obs)
    return wrapper