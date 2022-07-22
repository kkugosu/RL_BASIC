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

GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.policy = policy.Policy(self.cont, self.updatedPG, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=100)
        self.optimizer = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr)
        self.softmax = nn.Softmax(dim=-1)

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH))
            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buff()
            self.writer.add_scalar("pg/loss", loss, i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a_index = self.converter.act2index(n_a, self.b_s).unsqueeze(axis=-1)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)

            t_p_o_softmax = self.softmax(self.updatedPG(t_p_o))
            t_p_weight = torch.gather(t_p_o_softmax, 1, t_a_index)
            weight = torch.log(t_p_weight)
            loss = -torch.matmul(weight, t_r)
            # [1,2,3] * [1,2,3] = [1,4,9]
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1
            print("loss = ", loss)

        return loss
