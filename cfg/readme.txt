This document supports for how to use the cfg.json
Using cfg.json, you can easily control the algorithm and environment configuration.

1. explain the argument of json file
    envName: the name of environment
    sSize: the state size, actually it can not restrictly represent state size.
    Count: the length of unity env episode.
    resolution: the degree of raycast angle.
    nAgent: the number of agents in the game.
    maxStack: the number of fail.
    imgMode: (deprecated)
    aSize: the action size,
    seed: (deprecated)
    fixedSigma:(deprecated)
    coeffAngV: coefficient which penalty the agent if the angular velocity is larger than 0.7.
    coeffMAngV: coefficient which penalty the agent as much as the magnitude of angular velocity
    coeffDDist: coefficient which reinforces the agent as much as the differecne between the previous and current distance.
    coeffInnerProduct: coefficient.
    objRewardN: the number of achieved goals
    nUpdateDist: the period updating previous position for shaping reward.
    
    agent:
    optim:

    updateStep: the length of segmentic episod.
    div : the division of the current samples
    lrFreq: (deprecated)
    rScaling: (deprecated)

    gamma: discount factor
    Lambda: GAE
    epoch: the number of updating
    epsilon: Hyper-paramter of PPO algorithm
    entropyCoeff: 


