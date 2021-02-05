import json
import torch
from SAC.Agent import AgentV2


if __name__ == "__main__":
    jsonFilePath = "./cfg/RealTrain.json"

    with open(jsonFilePath) as file:
        json_dict = json.load(file)
        agentDict = json_dict['Agent']
    
    x = AgentV2(agentDict)

    rstate = torch.zeros((32, 8))
    lidarpt = torch.zeros((32, 1, 360))
    image = torch.zeros((32, 1, 96, 96))

    y = x.forward((rstate, lidarpt, image))
    print("here")
