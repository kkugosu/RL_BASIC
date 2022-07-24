import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch import nn


class Policy:
    def __init__(self, policy, model, envname):
        self.policy = policy
        self.model = model
        self.envname = envname
        self.converter = converter.Converter(self.envname)
        self.softmax = nn.Softmax(dim=-1)

    def select_action(self, n_p_o):
        t_p_o = torch.tensor(n_p_o, device=device, dtype=torch.float32)
        if self.policy == "DQN":
            if random.random() < 0.98:
                with torch.no_grad():
                    t_p_qsa = self.model(t_p_o)
                t_a_index = torch.argmax(t_p_qsa)
                n_a = self.converter.index2act(t_a_index, 1)
            else:
                n_a = self.converter.rand_act()
            return n_a

        elif self.policy == "PG":

            with torch.no_grad():
                probability = self.model(t_p_o)

            t_a_index = torch.multinomial(probability, 1)
            n_a = self.converter.index2act(t_a_index.squeeze(-1), 1)
            return n_a

        elif ((self.policy == "AC")
              | (self.policy == "TRPO")
              | (self.policy == "PPO")
              | (self.policy == "SAC")):

            with torch.no_grad():
                probability = self.model(t_p_o)

            t_a_index = torch.multinomial(probability, 1)
            n_a = self.converter.index2act(t_a_index.squeeze(-1), 1)
            return n_a

        elif self.policy == "DDPG":
            t_a = self.model(t_p_o)
            n_a = t_a.cpu().numpy()
            return n_a

        else:
            print("model name error")
            return None
