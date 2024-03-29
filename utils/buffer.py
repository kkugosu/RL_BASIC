import torch
import random
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMA = 0.98


class Simulate:

    def __init__(self, env, policy, step_size, done_penalty, name):
        self.env = env
        self.policy = policy
        self.step_size = step_size
        self.done_penalty = done_penalty
        self.performance = 0
        self.policy_name = name

    def renewal_memory(self, capacity, dataset, dataloader, rew):
        total_num = 0
        pause = 0
        failure = 1
        traj_l = 100

        while len(dataset) != 0:
            dataset.pop()
        if traj_l != 0:
            _index = 0
            circular = 1
            while total_num < capacity:

                n_p_s = self.env.reset()
                t = 0
                while t < capacity - total_num: # if pg, gain accumulate

                    with torch.no_grad():
                        n_a = self.policy.select_action(n_p_s)
                    # n_s, n_r, n_i = self.env.step(n_a[_index*self.a_l:(_index+1)*self.a_l])
                    n_s, n_r, n_i = self.env.step(n_a * 3)
                    if (t+1) == traj_l:
                        done = 1
                    else:
                        done = 0
                    # self.dataset.push(n_p_s, n_a, n_s, t, done, _index)
                    dataset.push(n_p_s, n_a, n_s, n_r, done)
                    # we need index.. so have to convert dataset
                    n_p_s = n_s
                    t = t + 1
                    if t == traj_l:

                        if circular == 1:
                            _index = _index + 1
                        total_num += t
                        t = 0
                        failure = failure + 1
                        break
        else:
            while total_num < capacity - pause:

                _index = 0
                n_p_s = self.env.reset()
                t = 0
                while t < capacity - total_num: # if pg, gain accumulate
                    with torch.no_grad():
                        n_a = self.policy.select_action(n_p_s)
                    n_s, n_r, n_d, n_i = self.env.step(n_a)
                    dataset.push(n_p_s, n_a, n_s, n_r, np.float32(n_d))
                    # we need index.. so have to convert dataset
                    n_p_s = n_s
                    t = t + 1
                    if n_d:
                        total_num += t
                        t = 0
                        failure = failure + 1
                        break
                pause = t

        pre_observation, action, observation, reward, done = next(iter(dataloader))

        total_performance = np.sum(reward)
        self.performance = total_performance / failure

        reward_2 = torch.zeros(len(pre_observation))
        if self.policy_name == "DDPG_bay":
            print("rew1 = ", reward)
            reward_2 = rew(pre_observation, action).cpu().detach().numpy()
            print("rew2 =", reward_2)
        reward = reward_2 + reward
        i = 0
        while i < len(pre_observation):

            dataset.push(pre_observation[i], action[i], observation[i], reward[i], done[i])
            i = i + 1

        self._reward_converter(dataset, dataloader)
        return self.performance

    def get_performance(self):
        return self.performance

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

