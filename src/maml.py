#!/usr/bin/env python

import utils
from agent import Agent
import solver as Solver

import numpy as np
import time

def maml_iteration(
    batch_size,
    theta,
    meta_lr,
    size=5,
    p_slip=0,
    terminal=None,
    debug=False,
    discount=0.7,
    draw=False
):
    # set-up mdp
    world, reward, terminal = utils.setup_mdp(size, p_slip, location=terminal)
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
    theta, maml_reward = utils.maxent(
        world,
        terminal,
        trajectories,
        theta
    )

    # Get a theta for an untrained init:
    theta_regular, reg_reward = utils.maxent(
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

    if draw: utils.plot_rewards(world, reward, expert_policy, trajectories, maml_reward, reg_reward)
    # update theta:
    theta = update_theta(theta_old, theta, meta_lr, debug)

    return theta, reward, maml_reward, reg_reward, world

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

def validate(size, optimal_policy, agent_policy):
    # compare the policies, remember that the terminal state's policy is unneeded
    error_num = sum(agent_policy != optimal_policy)
    return error_num / size**2

def calc_rewards(world, gt_reward, maml_reward, reg_reward, size, discount, debug=False):
     # optimal policy:
    optimal_policy = Solver.optimal_policy(world, gt_reward, discount)
    maxent_policy = Solver.optimal_policy(world, maml_reward, discount)
    reg_maxent_policy = Solver.optimal_policy(world, reg_reward, discount)

    # validate
    policy_score = validate(size, optimal_policy, maxent_policy)
    reg_policy_score = validate(size, optimal_policy, reg_maxent_policy)
    if debug:
        print("Maxent policy Score: {0}    :    Regulary policy score: {1}".format(
            policy_score, reg_policy_score
        ))
    validation_score = sum((maml_reward - gt_reward)**2)
    regular_score = sum((reg_reward - gt_reward)**2)
    return validation_score, regular_score, policy_score, reg_policy_score


def maml(N=100, batch_size=20, meta_lr=0.1, size=5, p_slip=0, terminal=None, debug=False, theta=None, discount=0.7, draw=False):
    data = {
        "thetas": [],
        "groundTruthReward": [],
        "mamlReward": [],
        "regularReward": [],
        "worlds": [],
         "validation_score": [],
         "regular_score": [],
         "policy_score": [],
         "reg_policy_score": []
    }

    for ind in range(N):
        startTime = time.time()
        # theta, gt_reward, maml_reward, reg_reward, world = maml_iteration(
        theta, gt_reward, maml_reward, reg_reward, world = maml_iteration(
            batch_size,
            theta,
            meta_lr,
            size,
            p_slip,
            terminal,
            debug,
            discount,
            draw
        )

        validation_score, regular_score, policy_score, reg_policy_score = calc_rewards(
            world,
            gt_reward,
            maml_reward,
            reg_reward,
            size,
            discount,
            debug
        )

        data["thetas"].append(theta.copy())
        data["groundTruthReward"].append(gt_reward)
        data["mamlReward"].append(maml_reward)
        data["regularReward"].append(reg_reward)
        data["worlds"].append(world)
        data["validation_score"].append(validation_score)
        data["regular_score"].append(regular_score)
        data["policy_score"].append(policy_score)
        data["reg_policy_score"].append(reg_policy_score)

        executionTime = (time.time() - startTime)

        if debug:
            print('Iteration #{0} execution time: {1} (sec) - \
                validation score: {2}, regular score: {3}'.
                format(
                    ind,
                    round(executionTime, 2),
                    validation_score,
                    regular_score
                )
            )

    return data

if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 8
    p_slip = 0.0
    N = 100
    batch_size = 20
    meta_lr = 0.1
    terminal = size**2 - 1
    debug = True
    data = maml(
        N=N,
        batch_size=batch_size,
        meta_lr=meta_lr,
        size=size,
        p_slip=p_slip,
        terminal=terminal,
        debug=debug
    )
    print('Theta: {0}'.format(data["thetas"][-1]))
    executionTime = (time.time() - startTime)
    print("mean validations per tenths:")
    print([np.round(np.mean(data["policy_score"][int(N / 10) * i :
        int(N / 10) * (i + 1)]), 2) for i in range(10)])
    print("Regular maxent:")
    print([np.round(np.mean(data["reg_policy_score"][int(N / 10) * i :
        int(N / 10) * (i + 1)]), 2) for i in range(10)])
    print('Total execution time: {0} (sec)'.format(executionTime))