from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torchvision.transforms import ToTensor, Lambda
from torch import nn
from NeuralNetwork import NN
from utils import buffer
import random
import torch.onnx as onnx
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.999


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.policy = policy.Policy(self.cont, self.updatedPG, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=100)
        self.optimizer = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr)
        self.softmax = nn.Softmax(dim=-1)

    def training(self):
        try:
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH_TEST))
        except:
            pass
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buff(self.t_i, self.b_s, self.optimizer, self.updatedPG)
            self.writer.add_scalar("loss", loss, i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH_TEST)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self, iteration, batch_size, optimizer, upd_model):
        i = 0
        while i < iteration:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            n_a_index = self.converter.act2index(n_a, batch_size).astype(np.int64)
            t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(device)
            print("trsh",np.shape(t_r))
            t_p_o_softmax = self.softmax(upd_model(t_p_o))
            state_action_values = torch.gather(upd_model(t_p_o_softmax), 1, t_a_index)
            print("sa",np.shape(state_action_values))
            weight = torch.log(state_action_values)
            loss = -torch.matmul(weight, t_r) #[1,2,3] * [1,2,3] = [1,4,9]
            print("wait")
            optimizer.zero_grad()
            loss.backward()
            for param in upd_model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            i = i + 1
            print("loss = ", loss)

        return loss

