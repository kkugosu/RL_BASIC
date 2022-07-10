import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# self.converter = converter.Converter(self.envname)




def rendering(self, forwardstep):

    pre_observation = self.env.reset()
    pre_observation = torch.tensor(pre_observation, device=device, dtype=torch.float32)
    total_num = 0
    t = 0
    while t < forwardstep - total_num:
        with torch.no_grad():
            basedqn_action = self.model(pre_observation)
        max_action = np.argmax(basedqn_action.cpu().numpy())
        action = self.converter.index2act(max_action, 1)
        observation, reward, done, info = self.env.step(action)
        pre_observation = torch.tensor(observation, device=device, dtype=torch.float32)
        self.env.render()
        t = t + 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            total_num += t
            t = 0
            break



