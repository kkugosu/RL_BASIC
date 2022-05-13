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

    def renewal_memory(self, renewal_capacity, dataset):
        total_num = 0
        pause = 0
        while total_num < renewal_capacity - pause:
            pre_observation = self.env.reset()
            pre_observation = torch.tensor(pre_observation, device=device, dtype=torch.float32)
            t = 0
            while t < renewal_capacity - total_num: #if pg, gain accumulate
                print(t)
                action = self.policy(pre_observation)
                observation, reward, done, info = self.env.step(action)
                np_pre_observation = pre_observation.cpu().numpy()
                dataset.push(np_pre_observation, action, observation, reward, np.float32(done))
                '''
                elif self.modelname == "PG":
                    if t == 0:
                        temp_preobs = np.array([np_pre_observation])
                        temp_action = np.array([action])
                        temp_obs = np.array([observation])
                        temp_reward = np.array([reward])
                    else:
                        temp_gamma = np.power(gamma, t)
                        temp_preobs = np.append(temp_preobs, np.array([pre_observation]), axis=0)
                        temp_action = np.append(temp_action, np.array([action]), axis=0)
                        temp_obs = np.append(temp_obs, np.array([observation]), axis=0)
                        temp_reward = np.append(temp_reward, np.array([reward*temp_gamma]), axis=0)
                else:
                    print("renewal_memory_error")
                '''
                pre_observation = torch.tensor(observation, device=device, dtype=torch.float32)
                t = t + 1
                '''
                if done:
                    temp_reward[t-1] = temp_reward[t-1] - np.float32(done)*temp_gamma
                    # if episode done in 5 step, t is 5
                    if self.modelname == "PG":
                        pg_step = 0
                        while pg_step < t:  # per step
                            reward_step = pg_step
                            gain = np.float32(0)
                            while reward_step < t:
                                gain = gain + temp_reward[reward_step]
                                reward_step = reward_step + 1
                            dataset.push(temp_preobs, temp_action, temp_obs, gain)
                            pg_step = pg_step + 1
                    # print("Episode finished after {} timesteps".format(t))
                    total_num += t
                    t = 0
                    break
                '''
            pause = t
        # print("load_memory_complete")
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


