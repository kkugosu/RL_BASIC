from control import BASE, policy
import gym
import torch
import numpy as np
import sys
from torch import nn
from NeuralNetwork import basic_nn
from utils import buffer
import random
import torch.onnx as onnx
GAMMA = 0.98


class DDPGPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = basic_nn.ValueNN(self.o_s, self.h_s, self.a_s).to(self.device)
        self.updatedDQN = basic_nn.ValueNN(self.o_s + self.a_s, self.h_s, 1).to(self.device)
        self.baseDQN = basic_nn.ValueNN(self.o_s + self.a_s, self.h_s, 1).to(self.device)
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.updatedPG, self.converter)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=self.e_trace, done_penalty=self.d_p)
        self.optimizer_p = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr, weight_decay=0.01)
        self.optimizer_q = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr, weight_decay=0.01)
        self.criterion = nn.MSELoss(reduction='mean')

    def get_policy(self):
        return self.policy

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH + "/1.pth"))
            self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH + "/2.pth"))
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()
            print("loading complete")
        else:
            pass
        i = 0
        torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/01.pth")
        torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/02.pth')
        while i < self.t_i:
            print(i)
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            pg_loss, dqn_loss = self.train_per_buff()
            self.writer.add_scalar("pg/loss", pg_loss, i)
            self.writer.add_scalar("dqn/loss", dqn_loss, i)
            self.writer.add_scalar("performance", self.buffer.get_performance(), i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/1.pth")
            torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/2.pth')
            if i == 100:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/11.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/21.pth')
            if i == 200:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/12.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/22.pth')
            if i == 300:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/13.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/23.pth')
            if i == 400:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/14.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/24.pth')
            if i == 500:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/15.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/25.pth')
            if i == 600:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/16.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/26.pth')
            if i == 700:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/17.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/27.pth')
            if i == 800:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/18.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/28.pth')
            if i == 900:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/19.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/29.pth')
            if i == 1000:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/110.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/210.pth')
            if i == 1100:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/111.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/211.pth')
            if i == 1200:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/112.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/212.pth')
            if i == 1300:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/113.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/213.pth')
            if i == 1400:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/114.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/214.pth')
            if i == 1500:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/115.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/215.pth')
            if i == 1600:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/116.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/216.pth')
            if i == 1700:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/117.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/217.pth')
            if i == 1800:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/118.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/218.pth')
            if i == 1900:
                torch.save(self.updatedPG.state_dict(), self.PARAM_PATH + "/119.pth")
                torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH + '/219.pth')

            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()

        for param in self.updatedDQN.parameters():
            print("----------dqn-------------")
            print(param)
        for param in self.updatedPG.parameters():
            print("----------pg--------------")
            print(param)

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buff(self):
        i = 0
        queue_loss = None
        policy_loss = None
        while i < self.m_i:

            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(self.device)
            t_a = torch.tensor(n_a, dtype=torch.float32).to(self.device)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(self.device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(self.device)
            #print("state = ",t_p_o.size())
            #print(t_p_o)
            #print("action = ",t_a.size())
            #print(t_a)
            #print("reward = ",t_o.size())
            #print(t_o)
            #print("reward = ", t_r.size())
            #print(t_r)
            dqn_input = torch.cat((t_p_o, t_a), dim=-1)
            t_p_qvalue = self.updatedDQN(dqn_input)
            dqn_input_req_grad = torch.cat((t_p_o, self.updatedPG(t_p_o)), dim=-1)

            policy_loss = - torch.mean(self.baseDQN(dqn_input_req_grad))
            t_trace = torch.tensor(n_d, dtype=torch.float32).to(self.device).unsqueeze(-1)

            with torch.no_grad():
                n_a_expect = self.policy.select_action(n_o)
                t_a_expect = torch.tensor(n_a_expect).to(self.device)
                dqn_input = torch.cat((t_o, t_a_expect), dim=-1)
                t_qvalue = self.baseDQN(dqn_input)*(GAMMA**t_trace) + t_r.unsqueeze(-1)

            queue_loss = self.criterion(t_p_qvalue, t_qvalue)

            self.optimizer_p.zero_grad()
            policy_loss.backward(retain_graph=True)
            for param in self.updatedPG.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_p.step()

            self.optimizer_q.zero_grad()
            queue_loss.backward()
            for param in self.updatedDQN.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer_q.step()

            i = i + 1

        print("loss1 = ", policy_loss)
        print("loss2 = ", queue_loss)

        return policy_loss, queue_loss
