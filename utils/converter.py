import random
import numpy as np
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.distributions.multivariate_normal import MultivariateNormal


class Converter:
    """
    torch index
    -> numpy action
    numpy action
    -> torch index
    """
    def __init__(self, env_name, a_s, precision):
        self.env_name = env_name
        self.a_s = a_s
        self.precision = precision
        self.gauge = 2.0/(self.precision - 1.0)

    def index2act(self, _input, batch):
        if self.env_name == "hope":
            if batch == 1:
                precision = torch.tensor(self.precision).to(DEVICE)
                div_1 = torch.div(_input, precision, rounding_mode='floor')
                div_2 = torch.div(div_1, precision, rounding_mode='floor')

                a_1 = _input % precision
                a_2 = div_1 % precision
                a_3 = div_2 % precision
                out = torch.tensor([a_1, a_2, a_3], device=DEVICE)*self.gauge - 1
            else:
                i = 0
                out = torch.zeros((batch, self.a_s), device=DEVICE)
                while i < batch:
                    precision = torch.tensor(self.precision).to(DEVICE)
                    div_1 = torch.div(_input, precision, rounding_mode='floor')
                    div_2 = torch.div(div_1, precision, rounding_mode='floor')
                    a_1 = _input % precision
                    a_2 = div_1 % precision
                    a_3 = div_2 % precision
                    out[i] = torch.tensor([a_1, a_2, a_3], device=DEVICE)*self.gauge - 1

                    i = i + 1

            return out.cpu().numpy()
        elif self.env_name == "cart":
            return _input.cpu().numpy()
        else:
            print("converter error")

    def act2index(self, _input, batch):
        if self.env_name == "hope":
            if batch == 1:
                _input = (_input+1)/self.gauge
                out = _input[2] * self.precision**2 + _input[1] * self.precision + _input[0]
            else:
                i = 0
                out = np.zeros(batch)
                while i < batch:
                    _input[i] = (_input[i]+1)/self.gauge
                    out[i] = _input[i][2] * self.precision**2 + _input[i][1] * self.precision + _input[i][0]
                    i = i + 1
            return torch.from_numpy(out).to(DEVICE).type(torch.int64)
        elif self.env_name == "cart":
            return torch.from_numpy(_input).to(DEVICE).type(torch.int64)
        else:
            print("converter error")

    def rand_act(self):
        if self.env_name == "hope":
            return (np.random.randint(self.precision, size=(self.a_s,))*self.gauge) - 1
        elif self.env_name == "cart":
            _a = np.random.randint(self.a_s, size=(1,))
            return _a[0]
        else:
            print("converter error")


class NAFPolicy:
    def __init__(self, sl, al, policy):
        self.sl = sl
        self.al = al
        self.policy = policy

    def prob(self, state):
        pre_psd, mean = torch.split(self.policy(state), [self.al**2, self.al], dim=-1)
        # print("prepse = ", pre_psd)
        # print("mean = ", mean)
        pre_psd = torch.reshape(pre_psd, (-1, self.al, self.al)).squeeze()
        pre_psd_trans = torch.transpose(pre_psd, -2, -1)
        psd = torch.matmul(pre_psd, pre_psd_trans)
        # psd = cov matrix
        # print(mean)
        # print(psd)
        try:
            normal = MultivariateNormal(mean, psd)
            return mean, psd, normal.sample()
        except ValueError:
            normal = MultivariateNormal(mean, psd + 0.0001)
            return mean, psd + 0.0001, normal.sample()
        # psd = psd no exception occered


