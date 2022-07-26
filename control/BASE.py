import gym
from torch.utils.tensorboard import SummaryWriter
from utils import converter
from utils import dataset, dataloader
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HOP_ACTION_SECTION = 5


class BasePolicy:
    """
    b_s batch_size
    ca capacity
    o_s observation space
    a_s action space
    h_s hidden space
    lr learning rate
    t_i training iteration
    cont control
    env_n environment name
    """
    def __init__(self,
                 b_s,
                 ca,
                 h_s,
                 lr,
                 t_i,
                 m_i,
                 cont,
                 env_n,
                 e_trace
                 ):
        self.b_s = b_s
        self.ca = ca
        self.h_s = h_s
        self.lr = lr
        self.t_i = t_i
        self.m_i = m_i
        self.cont = cont
        self.env_n = env_n
        self.e_trace = e_trace
        self.device = DEVICE

        self.PARAM_PATH = 'Parameter/' + self.env_n + self.cont
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = 'Parameter/' + self.env_n + self.cont + '_test'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

        if self.env_n == "cart":
            self.env = gym.make('CartPole-v1')
        elif self.env_n == "hope":
            self.env = gym.make('Hopper-v3')
        else:
            self.env = gym.make('Hopper-v3')

        self.o_s = len(self.env.observation_space.sample())
        self.a_s = len(self.env.action_space.sample())
        print("STATE_SIZE(input) = ", self.o_s)

        if self.env_n == "cart":
            self.a_s = len(self.env.action_space.sample())
            self.a_index_s = len(self.env.action_space.sample())
        else:
            self.a_s = len(self.env.action_space.sample())
            self.a_index_s = HOP_ACTION_SECTION ** self.a_s
        print("ACTION_SIZE(output) = ", self.a_s)
        
        self.data = dataset.SimData(capacity=self.ca)
        self.dataloader = dataloader.CustomDataLoader(self.data, batch_size=self.b_s)
        self.converter = converter.Converter(self.env_n, self.a_s, HOP_ACTION_SECTION)
        self.writer = SummaryWriter('Result/' + self.env_n + '/' + self.cont)
