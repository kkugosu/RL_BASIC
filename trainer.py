import torch
from torch import nn
from utils import converter
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Train:
    def __init__(self, env_name, data_loader, model_name):
        self.converter = converter.Converter(env_name)
        self.data_loader = data_loader
        self.model_name = model_name
        self.softmax = nn.Softmax(dim=-1)

    def training(self, iteration, batch_size, *model, optimizer):
        GAMMA = 0.999
        i = 0
        loss = None
        upd_model = None
        base_model = None

        if self.model_name == "DQN":
            upd_model = model[0]
            base_model = model[1]
            while i < iteration:
                # print(i)
                pre_observation, action, observation, reward, done = next(iter(self.data_loader))
                action_idx = self.converter.act2index(action, batch_size).astype(np.int64)
                action_idx = torch.from_numpy(action_idx).to(device).unsqueeze(axis=-1)
                pre_obs_to_cuda = torch.tensor(pre_observation, dtype=torch.float32).to(device)
                state_action_values = torch.gather(upd_model(pre_obs_to_cuda), 1, action_idx)
                obs_to_cuda = torch.tensor(observation, dtype=torch.float32).to(device)
                reward_to_cuda = torch.tensor(reward, dtype=torch.float32).to(device)
                criterion = nn.MSELoss()
                with torch.no_grad():
                    nextobs = base_model(obs_to_cuda)
                    expected_state_action_values = GAMMA * torch.argmax(nextobs, dim=1) + reward_to_cuda
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(axis=-1))
                optimizer.zero_grad()
                loss.backward()
                for param in upd_model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                i = i + 1
                print("loss = ", loss)

        elif self.model_name == "PG":
            upd_model = model[0]
            while i < iteration:
                # print(i)
                pre_observation, action, observation, reward, done = next(iter(self.data_loader))
                action_idx = self.converter.act2index(action, batch_size).astype(np.int64)
                action_idx = torch.from_numpy(action_idx).to(device).unsqueeze(axis=-1)
                pre_obs_to_cuda = torch.tensor(pre_observation, dtype=torch.float32).to(device)
                reward_to_cuda = torch.tensor(reward, dtype=torch.float32).to(device)
                pre_obs_softmax = self.softmax(upd_model(pre_obs_to_cuda))
                state_action_values = torch.gather(upd_model(pre_obs_softmax), 1, action_idx)
                weight = torch.log(state_action_values)
                loss = -weight * reward_to_cuda
                print("wait")
                optimizer.zero_grad()
                loss.backward()
                for param in upd_model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                i = i + 1
                print("loss = ", loss)
        else:
            print("model input error")
        return loss

