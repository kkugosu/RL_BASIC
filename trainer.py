import torch
from torch import nn
from utils import converter
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Train:
    def __init__(self, envname, dataloader, updmodel1, basemodel2):
        self.converter = converter.Converter(envname)
        self.dataloader = dataloader
        self.updmodel1 = updmodel1
        self.basemodel2 = basemodel2

    def training(self, iteration, batchsize, optimizer):
        GAMMA = 0.999
        i = 0
        while i < iteration:
            # print(i)
            pre_observation, action, observation, reward = next(iter(self.dataloader))

            action_idx = self.converter.act2index(action, batchsize).astype(np.int64)
            action_idx = torch.from_numpy(action_idx).to(device).unsqueeze(axis=-1)

            state_action_values = torch.gather(self.updmodel1(pre_observation), 1, action_idx)

            obs_to_cuda = torch.tensor(observation, dtype=torch.float32).to(device)
            reward_to_cuda = torch.tensor(reward, dtype=torch.float32).to(device)

            with torch.no_grad():
                nextobs = self.basemodel2(obs_to_cuda)
                expected_state_action_values = GAMMA * torch.argmax(nextobs, dim=1) + reward_to_cuda

            criterion = nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(axis=-1))

            optimizer.zero_grad()
            loss.backward()
            for param in self.updmodel1.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # for p in updatedDQN.parameters():
            #    print("params = ", p.grad)
            i = i + 1

        print("loss = ", loss)
        # print("train complete iter = ", iteration)
        return loss

