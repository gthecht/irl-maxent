#!/usr/bin/env python

import utils

import numpy as np
import matplotlib.pyplot as plt
import time

def maml_iteration(batch_size, theta, size=5, p_slip=0):
    # set-up mdp
    world, reward, terminal = utils.setup_mdp(size, p_slip)
    # get expert trajectories
    trajectories, expert_policy = utils.generate_trajectories(
        world,
        reward,
        terminal,
        batch_size
    )
    # optimize with maxent
    theta_maxent, reward_maxent = utils.maxent(
        world,
        terminal,
        trajectories
    )

    # optimize with maxent - causal
    # theta_maxcausal, reward_maxcausal = utils.maxent_causal(
    #     world,
    #     terminal,
    #     trajectories
    # )
    return theta_maxent, reward_maxent

def maml(N=100, batch_size=20, meta_lr=0.1, size=5, p_slip=0, theta=None):
    for ind in range(N):
        startTime = time.time()
        phi, reward = maml_iteration(batch_size, theta, size, p_slip)
        if theta is None:
            theta = phi
        else:
            theta = theta + meta_lr * (phi - theta)
        executionTime = (time.time() - startTime)
        print('Iterataion #{0} execution time: {1} (sec) - phi sum: {2}'.
              format(ind, round(executionTime, 2), np.abs(phi - theta).sum()))
    return theta

if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 5
    p_slip = 0.0
    N=100
    batch_size=20
    meta_lr=0.1
    theta = maml(N, batch_size, meta_lr, size, p_slip)
    print('Theta: {0}'.format(theta))
    executionTime = (time.time() - startTime)
    print('Total execution time: {0} (sec)'.format(executionTime))