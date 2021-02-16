#!/usr/bin/env python

import utils
from agent import Agent
import solver as Solver

import numpy as np
import matplotlib.pyplot as plt
import time

def maml_iteration(
    batch_size,
    theta,
    meta_lr,
    size=5,
    p_slip=0,
    discount=0.7
):
    # set-up mdp
    world, reward, terminal = utils.setup_mdp(size, p_slip)
    # get expert trajectories
    trajectories, expert_policy = utils.generate_trajectories(
        world,
        reward,
        terminal,
        n_trajectories=batch_size,
        discount=discount
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

    # update theta:
    theta = update_theta(theta, theta_maxent, meta_lr)
    agent = Agent(size=size, max_steps=max_steps)

    # optimal policy:
    optimal_policy = Solver.optimal_policy(world, reward, discount)

    # validate
    validation_score = validate(agent, world, terminal, theta, size, optimal_policy)


    return theta, theta_maxent, validation_score

def update_theta(theta, phi, meta_lr):
    """
    Update theta
    """

    if theta is None:
        theta = phi
    else:
        theta = theta + meta_lr * (phi - theta)
    return theta

def validate(agent, world, terminal, theta, size, optimal_policy):
    # The ground agent's policy:
    agent_policy = agent.get_policy(theta, size)
    # compare the policies, remember that the terminal state's policy is unneeded
    error_num = sum(agent_policy != optimal_policy)
    return error_num / size**2


def maml(N=100, batch_size=20, meta_lr=0.1, size=5, p_slip=0, max_steps=100, theta=None):
    validation_scores = np.zeros(N)
    for ind in range(N):
        startTime = time.time()
        theta, phi, validation_score = maml_iteration(
            batch_size,
            theta,
            meta_lr,
            size,
            p_slip
        )

        validation_scores[ind] = validation_score
        executionTime = (time.time() - startTime)
        print('Iterataion #{0} execution time: {1} (sec) - validation score: {2}'.
              format(ind, round(executionTime, 2), validation_score))
    return theta, validation_scores

if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 5
    p_slip = 0.0
    max_steps = 100
    N = 1000
    batch_size = 20
    meta_lr = 0.1
    theta, validation_scores = maml(N, batch_size, meta_lr, size, p_slip, max_steps)
    print('Theta: {0}'.format(theta))
    executionTime = (time.time() - startTime)
    print('Total execution time: {0} (sec)'.format(executionTime))