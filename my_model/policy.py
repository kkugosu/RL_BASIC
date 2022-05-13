import torch
import random
import numpy as np
from utils import converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#


class Policy:
    def __init__(self, modelname, model, envname):
        self.modelname = modelname
        self.model = model
        self.envname = envname
        self.converter = converter.Converter(self.envname)

    def select_action(self, pre_observation):
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
            my_action = torch.multinomial(basedqn_action, 1)
            my_action.cpu().numpy()
            action = self.converter.index2act(my_action, 1)
            return action

        else:
            print("model name error")
            return None
