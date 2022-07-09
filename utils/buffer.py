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
                action = self.policy.select_action(pre_observation)
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

    def __init__(self, step_size, dataloader):
        self.step_size = step_size
        self.dataloader = dataloader

    def reward_converter(self, dataset):
        t = 0
        pre_observation, action, observation, reward, done = next(iter(self.dataloader))
        # cal per trajectary to_end length ex) 4 3 2 1 6 5 4 3 2 1
        # set step to upper bound ex) step = 5 ->  4 3 2 1 5 5 4 3 2 1
        global_index = len(done) - 1
        local_index = 0
        done_penalty = 1
        while 0 <= global_index:
            if done[global_index] == 1:
                local_index = 1
                done[global_index] = local_index
                reward[global_index] -= done_penalty
                print("reset")
            else:
                local_index = local_index + 1
                if local_index > self.step_size:
                    local_index = self.step_size
                done[global_index] = local_index
            global_index = global_index - 1
        # cal newreward per state-action pair
        gamma = 0.9
        global_index = 0
        while global_index < len(done):
            observation[global_index] = observation[global_index + done[global_index] - 1]
            # change observation to last step indexed observation state
            local_index = 1
            while local_index < done[global_index]:
                tmp = reward[global_index + local_index] * gamma ** local_index
                reward[global_index] += tmp
                local_index = local_index + 1
            global_index += 1
        global_index = 0
        while global_index < len(done):
            dataset.push(np_pre_observation[global_index], action[global_index], observation[global_index],
                         reward[global_index], np.float32(done[global_index]))
            global_index += 1
        return dataset



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


