from control.base import BasePolicy
import gym
import torch
import numpy as np
import sys
from torchvision.transforms import ToTensor, Lambda
from torch import nn
from my_model import Network, policy
from utils import dataset, dataloader, buffer
import trainer
import random
import torch.onnx as onnx
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PGPolicy(BasePolicy):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.updatedPG = Network.SimpleNN(self.o_s, self.h_s, self.a_s).to(device)
        self.updatedPG.load_state_dict(torch.load(self.PARAM_PATH_TEST))
        self.policy = policy.Policy(self.cont, self.updatedPG, self.env_n)
        self.buffer = buffer.Simulate(self.env, self.policy)
        self.data = dataset.SimData(capacity=self.ca)
        self.buffer.renewal_memory(self.ca, self.data)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)
        self.train = trainer.Train(self.env_n, self.dataloader, self.cont)
        self.optimizer = torch.optim.SGD(self.updatedPG.parameters(), lr=self.lr)
        self.writer = SummaryWriter('RLresult/' + self.env_n + '/' + self.cont)

    def training(self):
        i = 0
        while i < self.t_i:
            i = i + 1
            self.buffer.renewal_memory(self.ca, self.data)
            loss = self.train.training(self.t_i, self.b_s, self.optimizer, self.updatedPG)
            self.writer.add_scalar("loss", loss, i)
            torch.save(self.updatedPG.state_dict(), self.PARAM_PATH_TEST)

        self.env.close()
        self.writer.flush()
        self.writer.close()
