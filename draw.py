import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from NeuralNetwork import basic_nn
os.environ['KMP_DUPLICATE_LIB_OK']='True'

HIDDEN_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


updatedPG = basic_nn.ValueNN(2, HIDDEN_SIZE, 2).to(DEVICE)
updatedDQN = basic_nn.ValueNN(4, HIDDEN_SIZE, 1).to(DEVICE)
PARAM_PATH = 'Parameter/' + "wallplane" + "DDPG"
updatedPG.load_state_dict(torch.load(PARAM_PATH + "/1.pth"))
updatedDQN.load_state_dict(torch.load(PARAM_PATH + "/2.pth"))

x = np.arange(0, 800, 1) - 400
y = np.arange(0, 800, 1) - 400

X, Y = np.meshgrid(x, y)
newX = np.expand_dims(X, axis=-1)
newY = np.expand_dims(Y, axis=-1)
X_a = np.ones_like(newX)
Y_a = np.ones_like(newY)

newXY = np.concatenate((newX, newY), -1)
newa = np.concatenate((X_a, Y_a), -1)
new = np.concatenate((newXY, newa), -1)
inp = torch.from_numpy(new).type(torch.float32)
out = updatedDQN(inp).squeeze()
npout = out.detach().numpy()
print("??")
sns.heatmap(npout)
plt.show()

newXY_t = torch.from_numpy(newXY)
