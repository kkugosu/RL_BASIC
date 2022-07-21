import base64
from varname import nameof
import gym
from control import DQN, PG, AC, BASE
from utils import render

if __name__ == "__main__":

    BATCH_SIZE = 10000
    CAPACITY = 10000
    TRAIN_ITER = 100
    MEMORY_ITER = 100
    HIDDEN_SIZE = 32
    learning_rate = 0.01

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
        elif envname == "hope":
            valid = 1
        else:
            print("error")

    valid = 0
    while valid == 0:
        print("enter RL control, {PG, DQN, ...}")
        control = input("->")
        if control == "PG":
            valid = 1
        elif control == "DQN":
            valid = 1
        else:
            print("error")

    print("enter batchsize")
    BATCH_SIZE = getinteger(BATCH_SIZE)

    print("enter memory capacity")
    CAPACITY = getinteger(CAPACITY)

    print("train iter per memory will be")
    TRAIN_ITER = getinteger(TRAIN_ITER)

    print("memory reset time will be")
    MEMORY_ITER = getinteger(MEMORY_ITER)

    print("enter HIDDEN_SIZE")
    HIDDEN_SIZE = getinteger(HIDDEN_SIZE)

    print("enter learning rate")
    learning_rate = getfloat(learning_rate)

    print("load enter 0 or 1")
    load_ = input("->")

    if control == PG:
        mechanism = PG.PGPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname)
        mechanism.training(load=load_)

    else:
        mechanism = DQN.DQNPolicy(BATCH_SIZE, CAPACITY, HIDDEN_SIZE,
                                  learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname)
        mechanism.training(load=load_)

    render.Render(BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate, TRAIN_ITER, MEMORY_ITER, control, envname)


