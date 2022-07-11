from control.base import BasePolicy
import torch
from my_model import Network, policy
from utils import dataset, dataloader, buffer
import trainer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import converter
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.999


class DQNPolicy(BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedDQN = Network.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.baseDQN = Network.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.updatedDQN.load_state_dict(torch.load(self.PARAM_PATH_TEST))
        self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
        self.baseDQN.eval()
        self.policy = policy.Policy(self.cont, self.baseDQN, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy)
        self.data = dataset.SimData(capacity=self.ca)
        self.buffer.renewal_memory(self.ca, self.data)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)
        self.train = trainer.Train(self.env_n, self.dataloader, self.cont)
        self.optimizer = torch.optim.SGD(self.updatedDQN.parameters(), lr=self.lr)
        self.converter = converter.Converter(self.env_n)
        self.writer = SummaryWriter('RLresult/' + self.env_n + '/' + self.cont)

    def training(self):
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data)
            loss = self.train_per_buf(self.t_i, self.b_s, self.optimizer, self.updatedDQN, self.baseDQN)
            self.writer.add_scalar("loss", loss, i)
            torch.save(self.updatedDQN.state_dict(), self.PARAM_PATH_TEST)
            self.baseDQN.load_state_dict(self.updatedDQN.state_dict())
            self.baseDQN.eval()

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def train_per_buf(self, iteration, batch_size, optimizer, *model):
        upd_model = model[0]
        base_model = model[1]
        i = 0
        while i < iteration:
            # print(i)
            n_p_o, n_a, n_o, n_r, n_d = next(iter(self.dataloader))
            n_a_index = self.converter.act2index(n_a, batch_size).astype(np.int64)
            t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
            t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
            t_p_qsa = torch.gather(upd_model(t_p_o), 1, t_a_index)
            t_o = torch.tensor(n_o, dtype=torch.float32).to(device)
            t_r = torch.tensor(n_r, dtype=torch.float32).to(device)
            criterion = nn.MSELoss()
            with torch.no_grad():
                t_p_qsa_ = base_model(t_o)
                t_p_qsa_ = torch.max(t_p_qsa_, dim=1)[0] * GAMMA
                t_p_qsa_ = t_p_qsa_ + t_r
            loss = criterion(t_p_qsa, t_p_qsa_.unsqueeze(axis=-1))
            optimizer.zero_grad()
            loss.backward()
            for param in upd_model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            i = i + 1
        return loss
