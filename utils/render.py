import torch
import random
import numpy as np
from NeuralNetwork import NN
from control import BASE, policy
from utils import converter
from control import policy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# self.converter = converter.Converter(self.envname)


class Render(BASE.BasePolicy):
    def __init__(self, *args):
        super().__init__(*args)
        self.MainNetwork = NN.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.MainNetwork.load_state_dict(torch.load(self.PARAM_PATH_TEST))
        self.policy = policy.Policy(self.cont, self.MainNetwork, self.env_n)
        n_p_o = self.env.reset()
        t = 0
        while t < 1000:
            n_a = self.policy.select_action(n_p_o)
            n_o, n_r, n_d, info = self.env.step(n_a)
            n_p_o = n_o
            self.env.render()
            t = t + 1
            if n_d:
                print("Episode finished after {} timesteps".format(t+1))
                break



