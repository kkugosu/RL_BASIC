from control import BASE, policy
import torch
from NeuralNetwork import NN
from utils import buffer
from torch import nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.98


class DQNPolicy(BASE.BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.MainNetwork = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.MainNetwork, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy, step_size=1)
        self.optimizer = torch.optim.SGD(self.MainNetwork.parameters(), lr=self.lr)

    def training(self, load=int(0)):

        if int(load) == 1:
            print("loading")
            self.MainNetwork.load_state_dict(torch.load(self.PARAM_PATH))
            print("loading complete")
        else:
            pass
        i = 0
        self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data, self.dataloader)
            loss = self.train_per_buf()
            print(loss)
            self.writer.add_scalar("loss", loss, i)
            torch.save(self.MainNetwork.state_dict(), self.PARAM_PATH)
            self.baseDQN.load_state_dict(self.MainNetwork.state_dict())
            self.baseDQN.eval()

        for p in self.baseDQN.parameters():
            print(p)
        for p in self.MainNetwork.parameters():
            print(p)
        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buf(self):
        upd_model = self.MainNetwork
        base_model = self.baseDQN
        i = 0
        while i < self.m_i:
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            n_a_index = self.converter.act2index(n_a, self.b_s).astype(np.int64)
            t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(device)

            t_p_qsa = torch.gather(upd_model(t_p_o), 1, t_a_index)
            criterion = nn.MSELoss()
            with torch.no_grad():
                t_p_qsa_ = base_model(t_o)
                t_p_qsa_ = torch.max(t_p_qsa_, dim=1)[0] * GAMMA
                t_p_qsa_ = t_p_qsa_ + t_r
            loss = criterion(t_p_qsa, t_p_qsa_.unsqueeze(axis=-1))
            self.optimizer.zero_grad()
            loss.backward()
            for param in upd_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            i = i + 1

        return loss
