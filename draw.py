import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from NeuralNetwork import basic_nn, bayesian_nn
os.environ['KMP_DUPLICATE_LIB_OK']='True'

HIDDEN_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


updatedPG = basic_nn.ValueNN(2, HIDDEN_SIZE, 2).to(DEVICE)
updatedDQN = basic_nn.ValueNN(4, HIDDEN_SIZE, 1).to(DEVICE)
dynamic = bayesian_nn.BayesianModel(4, HIDDEN_SIZE, 2).to(DEVICE)
PARAM_PATH = 'Parameter/' + "wallplane" + "TEST"
updatedPG.load_state_dict(torch.load(PARAM_PATH + "/1.pth"))
# updatedDQN.load_state_dict(torch.load(PARAM_PATH + "/2.pth"))
dynamic.load_state_dict(torch.load(PARAM_PATH + "/3.pth"))

x = np.arange(0, 200, 1) - 100
y = np.arange(0, 200, 1) - 100

X, Y = np.meshgrid(x, y)
newX = np.expand_dims(X, axis=-1)
newY = np.expand_dims(Y, axis=-1)
X_a = np.zeros_like(newX)
Y_a = np.ones_like(newY)

newXY = np.concatenate((newX, newY), -1)
newa = np.concatenate((X_a, Y_a), -1)
new = np.concatenate((newXY, newa), -1)
print("sg", np.shape(new))
inp = torch.from_numpy(new).type(torch.float32).to(DEVICE)
print(inp[0])
print(inp[1])
print(inp[2])
print(inp[3])
print(inp[4])
print("inp")
# out = updatedDQN(inp).squeeze()
# npout = out.detach().cpu().numpy()
print("??")
# sns.heatmap(npout)
# plt.show()
total_state_x = None
total_state_y = None
i = 0
while i < 10:
    print(i)
    new_state = dynamic(inp).squeeze()
    print(newXY[:, :, -1])
    new_state_x, new_state_y = torch.split(new_state, dim=-1, split_size_or_sections=[1, 1])
    print(new_state_y.squeeze())

    if i == 0:
        total_state_x = new_state_x
        total_state_y = new_state_y
    else:
        total_state_x = torch.cat((total_state_x, new_state_x), -1)
        total_state_y = torch.cat((total_state_y, new_state_y), -1)
    i = i + 1

sorted_state_x, _ = torch.sort(total_state_x, dim=-1)
sorted_state_y, _ = torch.sort(total_state_y, dim=-1)

re_1 = sorted_state_x[:, :, -1] - sorted_state_x[:, :, 0]
print(re_1)
re_2 = sorted_state_y[:, :, -1] - sorted_state_y[:, :, 0]
print(re_2)
unc = re_1 + re_2
print(unc)
n_unc = unc.cpu().detach().numpy()

sns.heatmap(n_unc)
plt.show()

