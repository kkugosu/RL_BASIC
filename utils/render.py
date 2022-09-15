import torch
import random
import numpy as np
from NeuralNetwork import basic_nn
from control import BASE
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Render(BASE.BasePolicy):
    def __init__(self, p, *args):
        super().__init__(*args)
        self.policy = p

    def rend(self):
        n_p_o = self.env.reset()
        t = 0
        total_performance = 0
        fail_time = 0
        while t < 1000:
            n_a = self.policy.select_action(n_p_o)
            n_o, n_r, info = self.env.step(n_a)
            total_performance = total_performance + n_r
            n_p_o = n_o
            self.env.render()
            t = t + 1
            if t % 200 == 0:
                print("Episode finished after {} timesteps".format(t))
                fail_time = fail_time + 1
                local = 0
                self.env.reset()
        print("performance = ", total_performance/fail_time)
