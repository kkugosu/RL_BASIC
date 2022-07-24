import base64
from varname import nameof
import gym
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

    def getinteger(integer):
        valid = 0
        while valid == 0:
            integer = input("->")
            try:
                int(integer)
                if float(integer).is_integer():
                    valid = 1
                    return int(integer)
                else:
                    print("enter integer")
            except ValueError:
                print("enter integer")


    def getfloat(float_):
        valid = 0
        while valid == 0:
            float_ = input("->")
            try:
                float(float_)
                valid = 1
                return float(float_)
            except ValueError:
                print("enter float")
    envname = None
    control = None

    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        envname = input("->")
        if envname == "cart":
            valid = 1
            print("we can't use DDPG")
        elif envname == "hope":
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
    HIDDEN_SIZE = getinteger(HIDDEN_SIZE)

    print("enter batchsize mostly 1000")
    BATCH_SIZE = getinteger(BATCH_SIZE)

    print("enter memory capacity mostly 1000")
    CAPACITY = getinteger(CAPACITY)

    print("memory reset time will be mostly 100")
    TRAIN_ITER = getinteger(TRAIN_ITER)

    print("train_iteration per memory mostly 20")
    MEMORY_ITER = getinteger(MEMORY_ITER)

    print("enter learning rate mostly 0.01")
    learning_rate = getfloat(learning_rate)

    print("load previous model 0 or 1")
    load_ = input("->")
    e_trace = 1

    if control == "PG":
        e_trace = 100
        mechanism = PG.PGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DQN":
        mechanism = DQN.DQNPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "DDPG":
        if envname == "hope":
            mechanism = DDPG.DDPGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                        learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
            mechanism.training(load=load_)
            policy = mechanism.get_policy()
        else:
            pass

    elif control == "TRPO":
        mechanism = TRPO.TRPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                    learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "PPO":
        mechanism = PPO.PPOPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "SAC":
        mechanism = SAC.SACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    elif control == "AC":
        mechanism = AC.ACPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
        mechanism.training(load=load_)
        policy = mechanism.get_policy()

    else:
        print("error")

    my_rend = render.Render(policy, BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate,
                            TRAIN_ITER, MEMORY_ITER, control, envname, e_trace)
    my_rend.rend()

