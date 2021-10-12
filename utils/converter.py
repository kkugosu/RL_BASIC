import random
import numpy as np


class Converter:
    def __init__(self, env):
        self.env = env

    def index2act(self, _input, batch):
        if self.env == "hope":
            if batch == 1:
                first_action = (_input % 5 / 2) - 1
                sec_action = ((_input % 25 - _input % 5) / 10) - 1
                third_action = ((_input - _input % 25) / 50) - 1
                out = np.array([first_action, sec_action, third_action])
            else:
                i = 0
                out = np.zeros((batch, 3))
                while i < batch:
                    first_action = (_input[i] % 5 / 2) - 1
                    sec_action = ((_input[i] % 25 - _input[i] % 5) / 10) - 1
                    third_action = ((_input[i] - _input[i] % 25) / 50) - 1
                    out[i] = np.array([first_action, sec_action, third_action])
                    i = i + 1
            return out
        elif self.env == "cart":
            return _input
        else:
            print("converter error")

    def act2index(self, _input, batch):
        if self.env == "hope":
            if batch == 1:
                _input = _input + 1
                _input = _input * 2
                out = _input[2] * 25 + _input[1] * 5 + _input[0]
            else:
                i = 0
                out = np.zeros(batch)
                while i < batch:
                    _input[i] = _input[i] + 1
                    _input[i] = _input[i] * 2
                    out[i] = _input[i][2] * 25 + _input[i][1] * 5 + _input[i][0]
                    i = i + 1
            return out
        elif self.env == "cart":
            return _input
        else:
            print("converter error")

    def rand_act(self):
        if self.env == "hope":
            return (np.random.randint(5, size=3) - 2)/2
        elif self.env == "cart":
            return np.random.randint(2)
        else:
            print("converter error")




