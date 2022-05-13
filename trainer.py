import torch
from torch import nn
from utils import converter
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Train:
    def __init__(self, env_name, data_loader, model_name):
        self.converter = converter.Converter(env_name)
        self.data_loader = data_loader
        self.model_name = model_name
        self.softmax = nn.Softmax(dim=-1)

    def training(self, iteration, batch_size, *model, optimizer):
        GAMMA = 0.999
        i = 0
        loss = None
        upd_model = None
        base_model = None

        if self.model_name == "DQN":
            upd_model = model[0]
            base_model = model[1]
        elif self.model_name == "PG":
            upd_model = model[0]
        else:
            print("model input error")

        while i < iteration:
            # print(i)
            pre_observation, action, observation, reward = next(iter(self.data_loader))
            action_idx = self.converter.act2index(action, batch_size).astype(np.int64)
            action_idx = torch.from_numpy(action_idx).to(device).unsqueeze(axis=-1)
            pre_obs_to_cuda = torch.tensor(pre_observation, dtype=torch.float32).to(device)
            state_action_values = torch.gather(upd_model(pre_obs_to_cuda), 1, action_idx)
            obs_to_cuda = torch.tensor(observation, dtype=torch.float32).to(device)
            reward_to_cuda = torch.tensor(reward, dtype=torch.float32).to(device)
            criterion = nn.MSELoss()

            if self.model_name == "DQN":
                with torch.no_grad():
                    nextobs = base_model(obs_to_cuda)
                    expected_state_action_values = GAMMA * torch.argmax(nextobs, dim=1) + reward_to_cuda
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(axis=-1))
            elif self.model_name == "PG":
                alpha = 0.1
                pre_obs_softmax = self.softmax(base_model(pre_obs_to_cuda))
                state_action_values = torch.gather(upd_model(pre_obs_softmax), 1, action_idx)
                weight = torch.log(state_action_values)
                loss = -alpha*weight*reward_to_cuda
                print("wait")
            else:
                print("training error")

            optimizer.zero_grad()
            loss.backward()
            for param in upd_model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # for p in updatedDQN.parameters():
            #    print("params = ", p.grad)
            i = i + 1

        print("loss = ", loss)
        # print("train complete iter = ", iteration)
        return loss

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