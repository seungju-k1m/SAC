import ray
import torch
from PPO.Agent import ppoAgent
from baseline.utils import jsonParser


if __name__ == "__main__":
    
    # load parser
    path = './cfg/LSTMTrain.json'
    parser = jsonParser(path)
    data = parser.loadParser()
    aData = parser.loadAgentParser()
    optimData = parser.loadOptParser()
    device = data['device']

    # init ray and torch_version
    ray.init()
    torch.set_default_dtype(torch.float64)

    # load parameter for agent
    
    entropyCoeff = data['entropyCoeff']
    epsilon = data['epsilon']
    lambda_ = data['lambda']
    initLogStd = torch.tensor(data['initLogStd']).to(device)
    finLogStd = torch.tensor(data['finLogStd']).to(device)
    annealingStep = data['annealingStep']
    LSTMNum = data['LSTMNum']

    ppo_ray = ray.remote(num_gpus=1)(ppoAgent)
    agent = ppo_ray.remote(
        aData,
        coeff=entropyCoeff,
        epsilon=epsilon,
        initLogStd=initLogStd,
        finLogStd=finLogStd,
        annealingStep=annealingStep,
        LSTMNum=LSTMNum
    )
    agent.to.remote(device)
    oldAgent = ppo_ray.remote(
        aData,
        coeff=entropyCoeff,
        epsilon=epsilon,
        initLogStd=initLogStd,
        finLogStd=finLogStd,
        annealingStep=annealingStep,
        LSTMNum=LSTMNum
    )
    oldAgent.to.remote(device)
    oldAgent.update.remote(agent)

    copyAgent = ppo_ray.remote(
        aData,
        coeff=entropyCoeff,
        epsilon=epsilon,
        initLogStd=initLogStd,
        finLogStd=finLogStd,
        annealingStep=annealingStep,
        LSTMNum=LSTMNum
    )

    copyAgent.to.remote(device)
    copyAgent.update.remote(agent)

    print("hello")
    