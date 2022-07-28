import torch
import random
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.98


class Simulate:

    def __init__(self, env, policy, step_size, done_penalty):
        self.env = env
        self.policy = policy
        self.step_size = step_size
        self.done_penalty = done_penalty
        
    def renewal_memory(self, capacity, dataset, dataloader):
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
        self._reward_converter(dataset, dataloader)

    def _reward_converter(self, dataset, dataloader):
        t = 0
        pre_observation, action, observation, reward, done = next(iter(dataloader))
        # cal per trajectary to_end length ex) 4 3 2 1 6 5 4 3 2 1
        # set step to upper bound ex) step = 5 ->  4 3 2 1 5 5 4 3 2 1
        global_index = len(done) - 1
        local_index = 0
        while 0 <= global_index:
            if done[global_index] == 1:
                local_index = 1
                done[global_index] = local_index
                reward[global_index] -= self.done_penalty
            else:
                local_index = local_index + 1
                if local_index > self.step_size:
                    local_index = self.step_size
                done[global_index] = local_index
            global_index = global_index - 1
        # cal newreward per state-action pair

        global_index = 0
        while global_index < len(done):
            observation[global_index] = observation[int(global_index + done[global_index] - 1)]
            # change observation to last step indexed observation state
            local_index = 1
            while local_index < done[global_index]:
                tmp = reward[global_index + local_index] * GAMMA ** local_index
                reward[global_index] += tmp
                local_index = local_index + 1
            global_index += 1
        global_index = 0
        while global_index < len(done):
            dataset.push(pre_observation[global_index], action[global_index], observation[global_index],
                         reward[global_index], np.float32(done[global_index]))
            global_index += 1
        return dataset

