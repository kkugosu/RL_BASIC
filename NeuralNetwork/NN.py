import torch
import numpy as np
from torchvision.transforms import ToTensor, Lambda
from torch import nn


class SimpleNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),

        )

    def forward(self, input_element):
        output = self.linear_relu_stack(input_element)
        return output

