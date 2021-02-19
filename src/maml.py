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
    debug=False,
    discount=0.7
):
    # set-up mdp
    world, reward, terminal = utils.setup_mdp(size, p_slip, location=size**2-1)
    # get expert trajectories
    trajectories, expert_policy = utils.generate_trajectories(
        world,
        reward,
        terminal,
        n_trajectories=batch_size,
        discount=discount
    )

    if theta is None: theta_old = None
    else: theta_old = theta.copy()
    # optimize with maxent
    theta, reward = utils.maxent(
        world,
        terminal,
        trajectories,
        theta
    )

    # Get a theta for an untrained init:
    theta_regular, _ = utils.maxent(
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
    theta = update_theta(theta_old, theta, meta_lr, debug)
    agent = Agent(size=size)

    # optimal policy:
    optimal_policy = Solver.optimal_policy(world, reward, discount)

    # validate
    validation_score = validate(agent, theta, size, optimal_policy)
    regular_score = validate(agent, theta_regular, size, optimal_policy)



    return theta, validation_score, regular_score

def update_theta(theta, phi, meta_lr, debug):
    """
    Update theta
    """

    # normalize phi
    # phi = phi / sum(phi)
    if theta is None: theta = np.ones_like(phi)
    theta = theta + meta_lr * (phi - theta)
    if debug: print("(Theta - Phi)^2: {0}".format(np.sum((phi - theta)**2)))
    return theta

def validate(agent, theta, size, optimal_policy):
    # The ground agent's policy:
    agent_policy = agent.get_policy(theta, size)
    # compare the policies, remember that the terminal state's policy is unneeded
    error_num = sum(agent_policy != optimal_policy)
    return error_num / size**2


def maml(N=100, batch_size=20, meta_lr=0.1, size=5, p_slip=0, debug=False, theta=None):
    validation_scores = np.zeros(N)
    reg_scores = np.zeros(N)
    for ind in range(N):
        startTime = time.time()
        theta, validation_score, validation_score_regular = maml_iteration(
            batch_size,
            theta,
            meta_lr,
            size,
            p_slip,
            debug
        )

        validation_scores[ind] = validation_score
        reg_scores[ind] = validation_score_regular
        executionTime = (time.time() - startTime)
        if debug:
            print('Iteration #{0} execution time: {1} (sec) - \
                validation score: {2}, regular score: {3}'.
                format(
                    ind,
                    round(executionTime, 2),
                    validation_score,
                    validation_score_regular
                )
            )
    return theta, validation_scores, reg_scores

if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 8
    p_slip = 0.0
    N = 100
    batch_size = 20
    meta_lr = 0.1
    debug = True
    theta, validation_scores, reg_scores = \
        maml(N, batch_size, meta_lr, size, p_slip, debug)
    print('Theta: {0}'.format(theta))
    executionTime = (time.time() - startTime)
    print("mean validations per tenths:")
    print([np.round(np.mean(validation_scores[int(N / 10) * i :
        int(N / 10) * (i + 1)]), 2) for i in range(10)])
    print("Regular maxent:")
    print([np.round(np.mean(reg_scores[int(N / 10) * i :
        int(N / 10) * (i + 1)]), 2) for i in range(10)])
    print('Total execution time: {0} (sec)'.format(executionTime))