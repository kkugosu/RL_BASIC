import gym
HOP_ACTION = 125
CART_ACTION = 2


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
                 cont,
                 env_n
                 ):
        self.b_s = b_s
        self.ca = ca
        self.h_s = h_s
        self.lr = lr
        self.t_i = t_i
        self.cont = cont
        self.env_n = env_n

        self.PARAM_PATH = self.cont + '_param/' + self.env_n + self.cont + '.pth'
        print("parameter path is " + self.PARAM_PATH)

        self.PARAM_PATH_TEST = self.cont + '_param/' + self.env_n + self.cont + '_test.pth'
        print("tmp parameter path is " + self.PARAM_PATH_TEST)

        if self.env_n == "cart":
            self.env = gym.make('CartPole-v1')
        elif self.env_n == "hope":
            self.env = gym.make('Hopper-v3')
        else:
            self.env = gym.make('Hopper-v3')

        self.o_s = len(self.env.observation_space.sample())
        print("STATE_SIZE(input) = ", self.o_s)

        if self.env_n == "cart":
            self.a_s = CART_ACTION
        else:
            self.a_s = HOP_ACTION
        print("ACTION_SIZE(output) = ", self.a_s)
