#!/usr/bin/env python

import gridworld as World
import maxent as Maxent
import plot as Plot
import trajectory as Trajectory
import solver as Solver
import optimizer as Optimizer

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Constants:
SIZE = 5
P_SLIP = 0

PHI_Y = np.random.rand(1)
PHI_X = np.random.rand(1)
FREQ_X = np.random.rand(1)
FREQ_Y = np.random.rand(1)
AMPLITUDE = -0.5

print("phi_x = {0}, phi_y = {1}, freq_x = {2}, freq_y = {3}".format(PHI_X, PHI_Y, FREQ_X, FREQ_Y))

def setup_mdp():
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = World.IcyGridWorld(size=SIZE, p_slip=P_SLIP)

    # set up the reward function
    reward = np.zeros(world.n_states)
    for ind in range(world.n_states):
        y_ind = round(ind / SIZE)
        x_ind = ind % SIZE
        reward[ind] = AMPLITUDE * 0.5 * (2 - math.sin(math.pi * FREQ_X * x_ind + PHI_X) - math.sin(math.pi * FREQ_Y * y_ind + PHI_Y))

    reward[-1] = 1.0
    reward[8] = 0.65
    # set up terminal states
    terminal = [SIZE * SIZE - 1]

    return world, reward, terminal


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 200
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    # generate trajectories
    value = Solver.value_iteration(world.p_transition, reward + AMPLITUDE, discount)
    policy = Solver.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = Trajectory.stochastic_policy_adapter(policy)
    tjs = list(Trajectory.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = World.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = Optimizer.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = Optimizer.ExpSga(lr=Optimizer.linear_decay(lr0=0.02))

    # actually do some inverse reinforcement learning
    reward = Maxent.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = World.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = Optimizer.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = Optimizer.ExpSga(lr=Optimizer.linear_decay(lr0=0.02))

    # actually do some inverse reinforcement learning
    reward = Maxent.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward


def main():
    startTime = time.time()
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    Plot.plot_state_values(ax, world, reward, **style)
    plt.draw()

    # generate "expert" trajectories
    trajectories, expert_policy = generate_trajectories(world, reward, terminal)

    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    Plot.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        Plot.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()

    # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(world, terminal, trajectories)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    Plot.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()

    # maximum casal entropy reinforcement learning (non-causal)
    reward_maxcausal = maxent_causal(world, terminal, trajectories)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    Plot.plot_state_values(ax, world, reward_maxcausal, **style)
    plt.draw()
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    plt.show()


if __name__ == '__main__':
    main()
