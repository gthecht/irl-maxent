#!/usr/bin/env python

import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O
import utils

import numpy as np
import matplotlib.pyplot as plt
import time

def maml_iteration(batch_size, theta):
    # set-up mdp
    world, reward, terminal = utils.setup_mdp()

    trajectories, expert_policy = utils.generate_trajectories(
        world,
        reward,
        terminal
    )
    theta_maxent, reward_maxent = utils.maxent(
        world,
        terminal,
        trajectories
    )
    # theta_maxcausal, reward_maxcausal = utils.maxent_causal(
    #     world,
    #     terminal,
    #     trajectories
    # )
    return theta_maxent, reward_maxent

def maml(N=100, batch_size=20, meta_lr=0.1, theta=O.Constant(1.0)):
    for ind in range(N):
        startTime = time.time()
        phi, reward = maml_iteration(batch_size, theta)
        theta = theta + meta_lr * (phi - theta)
        executionTime = (time.time() - startTime)
        print('Iterataion #{0} execution time: {1} (sec)'.format(ind, executionTime))
    return theta

if __name__ == '__main__':
    startTime = time.time()
    N=100
    batch_size=20
    meta_lr=0.1
    theta=O.Constant(1.0)
    theta = maml(N, batch_size, meta_lr, theta)
    print('Theta: {0}'.format(theta))
    executionTime = (time.time() - startTime)
    print('Total execution time: {0} (sec)'.format(executionTime))