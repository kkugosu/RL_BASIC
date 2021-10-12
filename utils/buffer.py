import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ManageMem:
    def __init__(self, model, modelname, env, envname):
        self.model = model
        self.env = env
        self.envname = envname
        self.modelname = modelname
        self.converter = converter.Converter(self.envname)

    def _select_act(self, pre_observation):
        if self.modelname == "DQN":
            if random.random() < 0.9:
                with torch.no_grad():
                    basedqn_action = self.model(pre_observation)
                max_action = np.argmax(basedqn_action.cpu().numpy())
                action = self.converter.index2act(max_action, 1)
            else:
                action = self.converter.rand_act()
            return action

        elif self.modelname == "PG":
            with torch.no_grad():
                basedqn_action = self.model(pre_observation)
            max_action = np.argmax(basedqn_action.cpu().numpy())
            action = self.converter.index2act(max_action, 1)
            return action

        else:
            print("model name error")
            return None

    def renewal_memory(self, renewal_capacity, dataset):
        total_num = 0
        pause = 0
        while total_num < renewal_capacity - pause:
            pre_observation = self.env.reset()
            pre_observation = torch.tensor(pre_observation, device=device, dtype=torch.float32)
            t = 0
            while t < renewal_capacity - total_num:
                t = t + 1
                action = self._select_act(pre_observation)
                observation, reward, done, info = self.env.step(action)
                np_pre_observation = pre_observation.cpu().numpy()
                dataset.push(np_pre_observation, action, observation, reward - np.float32(done))
                pre_observation = torch.tensor(observation, device=device, dtype=torch.float32)
                if done:
                    # print("Episode finished after {} timesteps".format(t+1))
                    total_num += t
                    t = 0
                    break
            pause = t
        # print("load_memory_complete")

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

