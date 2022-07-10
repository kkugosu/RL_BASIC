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
                n_p_o, n_a, n_o, n_r, n_d = next(iter(self.data_loader))
                n_a_index = self.converter.act2index(n_a, batch_size).astype(np.int64)
                t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
                t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
                t_p_qsa = torch.gather(upd_model(t_p_o), 1, t_a_index)
                t_o = torch.tensor(n_o, dtype=torch.float32).to(device)
                t_r = torch.tensor(n_r, dtype=torch.float32).to(device)
                criterion = nn.MSELoss()
                with torch.no_grad():
                    t_p_qsa_ = base_model(t_o)
                    t_p_qsa_ = torch.max(t_p_qsa_, dim=1)*GAMMA
                    t_p_qsa_ = t_p_qsa_ + t_r
                loss = criterion(t_p_qsa, t_p_qsa_.unsqueeze(axis=-1))
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
                n_p_o, n_a, n_o, n_r, n_d  = next(iter(self.data_loader))
                n_a_index = self.converter.act2index(n_a, batch_size).astype(np.int64)
                t_a_index = torch.from_numpy(n_a_index).to(device).unsqueeze(axis=-1)
                t_p_o = torch.tensor(n_p_o, dtype=torch.float32).to(device)
                t_r = torch.tensor(n_r, dtype=torch.float32).to(device)
                t_p_o_softmax = self.softmax(upd_model(t_p_o))
                state_action_values = torch.gather(upd_model(t_p_o_softmax), 1, t_a_index)
                weight = torch.log(state_action_values)
                loss = -weight * t_r
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

