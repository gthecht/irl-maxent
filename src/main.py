#!/usr/bin/env python

import plot as P

import matplotlib.pyplot as plt
import time

import utils

def main():
    startTime = time.time()
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = utils.setup_mdp()

    fig = plt.figure()
    # show our original reward
    # ax = plt.figure(num='Original Reward').add_subplot(111)
    ax = fig.add_subplot(221)
    P.plot_state_values(ax, world, reward, **style)
    plt.draw()
    plt.title("Original Reward")

    # generate "expert" trajectories
    trajectories, expert_policy = utils.generate_trajectories(world, reward, terminal)

    # show our expert policies
    # ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    ax = fig.add_subplot(222)
    P.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()
    plt.title("Expert Trajectories and Policy")

    # maximum entropy reinforcement learning (non-causal)
    theta_maxent, reward_maxent = utils.maxent(world, terminal, trajectories)

    print("Theta maxEnt: \n{0}".format(theta_maxent))
    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    ax = fig.add_subplot(223)
    P.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()
    plt.title("MaxEnt Reward")

    # maximum casal entropy reinforcement learning (non-causal)
    theta_maxcausal, reward_maxcausal = utils.maxent_causal(world, terminal, trajectories)

    print("Theta maxEnt - causal: \n{0}".format(theta_maxcausal))
    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    ax = fig.add_subplot(224)
    P.plot_state_values(ax, world, reward_maxcausal, **style)
    plt.draw()
    plt.title("MaxEnt Reward (Causal)")

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    plt.show()


if __name__ == '__main__':
    main()
