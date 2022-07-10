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
            n_p_o = self.env.reset()
            t = 0
            while t < memory_capacity - total_num: #if pg, gain accumulate
                n_a = self.policy.select_action(n_p_o)

                n_o, n_r, n_d, n_i = self.env.step(n_a)
                dataset.push(n_p_o, n_a, n_o, n_r, np.float32(n_d))
                n_p_o = n_o
                t = t + 1
                if n_d:
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

