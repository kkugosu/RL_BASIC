import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#


class Policy:
    def __init__(self, policy, model, envname):
        self.policy = policy
        self.model = model
        self.envname = envname
        self.converter = converter.Converter(self.envname)

    def select_action(self, n_p_o):
        t_p_o = torch.tensor(n_p_o, device=device, dtype=torch.float32)
        if self.policy == "DQN":
            if random.random() < 0.9:
                with torch.no_grad():
                    t_p_qsa = self.model(t_p_o)
                n_a_index = np.argmax(t_p_qsa.cpu().numpy())
                n_a = self.converter.index2act(n_a_index, 1)
            else:
                n_a = self.converter.rand_act()
            return n_a

        elif self.policy == "PG":
            with torch.no_grad():
                t_p_qsa = self.model(t_p_o)
            t_a_index = torch.multinomial(t_p_qsa.exp(), 1)
            n_a_index = t_a_index.cpu().numpy()
            n_a = self.converter.index2act(n_a_index, 1)
            return n_a

        else:
            print("model name error")
            return None
