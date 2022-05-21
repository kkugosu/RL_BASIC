import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# self.converter = converter.Converter(self.envname)


class Simulate:

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def renewal_memory(self, capacity, dataset):
        total_num = 0
        pause = 0
        memory_capacity = capacity
        while total_num < memory_capacity - pause:
            pre_observation = self.env.reset()
            pre_observation = torch.tensor(pre_observation, device=device, dtype=torch.float32)
            t = 0
            while t < memory_capacity - total_num: #if pg, gain accumulate
                print(t)
                action = self.policy(pre_observation)
                observation, reward, done, info = self.env.step(action)
                np_pre_observation = pre_observation.cpu().numpy()
                dataset.push(np_pre_observation, action, observation, reward, np.float32(done))

                pre_observation = torch.tensor(observation, device=device, dtype=torch.float32)
                t = t + 1
                if done:
                    total_num += t
                    t = 0
                    break
            pause = t

class RewardConverter:

    def __init__(self, step_size):
        self.step_size = step_size

    def reward_converter(self, dataloader):
        t = 0
        pre_observation, action, observation, reward, done = next(iter(self.data_loader))
        #extract done index
        #cal per trajectary to_end length ex) 4 3 2 1 6 5 4 3 2 1
        #set step to upper bound ex) step = 5 ->  4 3 2 1 5 5 4 3 2 1
        #cal newreward per state-action pair
        #change observation to last step indexed observation state
        return tmp_dataset



'''
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
'''


