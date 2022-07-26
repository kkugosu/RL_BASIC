from control import DQN, PG, AC, DDPG, SAC, TRPO, PPO
from utils import render

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    HIDDEN_SIZE = 32
    learning_rate = 0.01
    policy = None

    def get_integer():
        _valid = 0
        while _valid == 0:
            integer = input("->")
            try:
                int(integer)
                if float(integer).is_integer():
                    _valid = 1
                    return int(integer)
                else:
                    print("enter integer")
            except ValueError:
                print("enter integer")

    def get_float():
        _valid = 0
        while _valid == 0:
            float_ = input("->")
            try:
                float(float_)
                _valid = 1
                return float(float_)
            except ValueError:
                print("enter float")

    env_name = None
    control = None

    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        env_name = input("->")
        if env_name == "cart":
            valid = 1
            print("we can't use DDPG")
        elif env_name == "hope":
            valid = 1
        else:
            print("error")

    valid = 0
    while valid == 0:
        print("enter RL control, {PG, DQN, AC, TRPO, PPO, DDPG, SAC}")
        control = input("->")
        if control == "PG":
            valid = 1
        elif control == "DQN":
            valid = 1
        elif control == "AC":
            valid = 1
        elif control == "TRPO":
            valid = 1
        elif control == "PPO":
            valid = 1
        elif control == "DDPG":
            valid = 1
        elif control == "SAC":
            valid = 1
        else:
            print("error")

    print("enter HIDDEN_SIZE mostly 32")
    HIDDEN_SIZE = get_integer()

    print("enter batchsize mostly 1000")
    BATCH_SIZE = get_integer()

    print("enter memory capacity mostly 1000")
    CAPACITY = get_integer()

    print("memory reset time will be mostly 100")
    TRAIN_ITER = get_integer()

    print("train_iteration per memory mostly 20")
    MEMORY_ITER = get_integer()

    print("enter learning rate mostly 0.01")
    learning_rate = get_float()

    print("load previous model 0 or 1")
    load_ = input("->")
    e_trace = 1

    if control == "PG":
        e_trace = 100
        mechanism = PG.PGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DQN":
        mechanism = DQN.DQNPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DDPG":
        if env_name == "hope":
            mechanism = DDPG.DDPGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                        learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
            mechanism.training(load=load_)
            policy = mechanism.get_policy()
        else:
            pass

    elif control == "TRPO":
        mechanism = TRPO.TRPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                    learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "PPO":
        mechanism = PPO.PPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "SAC":
        mechanism = SAC.SACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "AC":
        mechanism = AC.ACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    else:
        print("error")

    my_rend = render.Render(policy, BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate,
                            TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace)
    my_rend.rend()
