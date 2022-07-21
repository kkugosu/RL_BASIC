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
GAMMA = 0.98


class PGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.updatedDQN = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.updatedPG, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=100)
        self.optimizer_p = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr)
        self.optimizer_q = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr)
        self.softmax = nn.Softmax(dim=-1)

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH + "/1"))
            self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH + "/2"))
            print("loading complete")
        else:
            pass
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buff()
            self.writer.add_scalar("loss", loss, i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/1")
            torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/2')
        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        while i < self.m_i:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            n_a_index = self.converter.act2index(n_a, self.b_s).astype(np.int64)
            t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(device)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(device)

            t_p_o_softmax = self.softmax(self.updatedPG(t_p_o))
            state_action_values = torch.gather(t_p_o_softmax, 1, t_a_index)
            DQN_weight = torch.gather(self.updatedDQN(t_p_o), 1, t_a_index)

            weight = torch.log(state_action_values)
            pgloss = -DQN_weight*weight
            # [1,2,3] * [1,2,3] = [1,4,9]
            with torch.no_grad():
                action = self.policy.select_action(n_o)
                n_a_index = self.converter.act2index(action, self.b_s).astype(np.int64)
                DQN_weight = torch.gather(self.baseDQN(t_o), 1, t_a_index)

                t_p_qsa_ = DQN_weight*GAMMA + t_r
            DQN_loss = DQN_weight - t_p_qsa_
            print("wait")

            self.optimizer_p.zero_grad()
            pgloss.backward()
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            DQN_loss.backward()
            for param in self.updatedDQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1
            print("loss = ", loss)

        return loss
