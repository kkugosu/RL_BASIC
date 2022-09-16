from control import DQN, PG, AC, DDPG, SAC, TRPO, PPO, SAC_conti
from utils import render
from simple_env import wallplane, plane, narrow
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    e_trace = 1
    precision = 5

    valid = 0
    while valid == 0:
        print("enter envname, {wallplane, plane, narrow}")
        env_name = "wallplane" #input("->")
        if env_name == "wallplane":
            valid = 1

        elif env_name == "plane":
            valid = 1

        elif env_name == "narrow":
            valid = 1

        else:
            print("error")

    """
    valid = 0
    while valid == 0:
        print("enter envname, {cartpole as cart, hoppper as hope}")
        env_name = input("->")
        if env_name == "cart":
            valid = 1
            print("we can't use DDPG")
        elif env_name == "hope":
            valid = 1
            print("enter hopper precision 3 or 5")
            precision = get_integer()
        else:
            print("error")
    """


    valid = 0
    while valid == 0:
        print("enter RL control, {PG, DQN, AC, TRPO, PPO, DDPG, SAC}")
        control = "DDPG" # input("->")
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
        elif control == "SAC_conti":
            valid = 1
        else:
            print("error")

    print("enter HIDDEN_SIZE recommend 32")
    HIDDEN_SIZE = 100 # get_integer()

    print("enter batchsize recommend 1000")
    BATCH_SIZE = 2000 # get_integer()

    print("enter memory capacity recommend 1000")
    CAPACITY = 2000 # get_integer()

    print("memory reset time recommend 100")
    TRAIN_ITER = 200 # get_integer()

    print("train_iteration per memory recommend 10")
    MEMORY_ITER = 1 # get_integer()

    print("enter learning rate recommend 0.01")
    learning_rate = 0.001 # get_float()

    print("enter eligibility trace step, if pg: 100")
    e_trace = 1 # get_integer()

    print("done penalty, if cartpole, recommend 10")
    done_penalty = 1 # get_integer()

    print("load previous model 0 or 1")
    load_ = 0 # input("->")
    mechanism = None
    arg_list = [BATCH_SIZE, CAPACITY, HIDDEN_SIZE, learning_rate,
                TRAIN_ITER, MEMORY_ITER, control, env_name, e_trace, precision, done_penalty]
    print(arg_list)
    if control == "PG":
        mechanism = PG.PGPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "DQN":
        mechanism = DQN.DQNPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "AC":
        mechanism = AC.ACPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "DDPG":
        if env_name == "cart":
            pass
        else:
            mechanism = DDPG.DDPGPolicy(*arg_list)
            mechanism.training(load=load_)
            # policy = mechanism.get_policy()

    elif control == "TRPO":
        mechanism = TRPO.TRPOPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "PPO":
        mechanism = PPO.PPOPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "SAC":
        mechanism = SAC.SACPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    elif control == "SAC_conti":
        mechanism = SAC_conti.SACPolicy(*arg_list)
        mechanism.training(load=load_)
        # policy = mechanism.get_policy()

    else:
        print("error")

    #mechanism.updatedPG.load_state_dict(torch.load(mechanism.PARAM_PATH + "/18.pth"))
    #mechanism.updatedDQN.load_state_dict(torch.load(mechanism.PARAM_PATH + "/28.pth"))

    policy = mechanism.get_policy()
    my_rend = render.Render(policy, *arg_list)
    my_rend.rend()
