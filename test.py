import torch
from SAC.Agent import AgentV1


if __name__ == "__main__":

    x = {
        "module01":{
          "netCat": "Res1D",
          "nBlock": 1,
          "iSize":32,
          "act": "relu",
          "BN": False,
          "linear":True,
          "input":[0]
        },
        "module02":{
          "netCat":"Cat",
          "input":[1, 2]
        },
        "module03":{
          "netCat": "MLP",
          "nLayer": 2,
          "fSize": [256, 1],
          "act": ["relu", "linear"],
          "BN": False,
          "iSize":3848,
          "output":True
        }
      }
    agent = AgentV1(x)
    a = AgentV1(x)
    agent.updateParameter(a, tau=0.05)
    optim = agent.buildOptim()

    inputs = (torch.zeros((1,1,120)), torch.ones(1,6), torch.zeros(1,2))
    x = agent.forward(inputs)
    print(1)