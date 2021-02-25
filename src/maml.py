#!/usr/bin/env python

import utils
import solver as Solver
import gridworld as World
import gaussianfit as Gfit

import numpy as np
import matplotlib.pyplot as plt

import time

def get_loss(size, world, gt_reward, reward, discount):
    # Calculate loss:
    optimal_policy_value = Solver.optimal_policy_value(world, gt_reward, discount)
    maxent_policy_value = Solver.optimal_policy_value(world, reward, discount)

    # validate
    loss = validate(size, world, optimal_policy_value, maxent_policy_value)
    return loss


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

    # optimize with maxent
    phi, phi_reward = utils.maxent(
        world,
        terminal,
        trajectories,
        theta
    )
    phi_loss = get_loss(size, world, reward, phi_reward, discount)

    # Get a theta for an untrained init:
    theta_regular, reg_reward = utils.maxent(
        world,
        terminal,
        trajectories
    )
    reg_loss = get_loss(size, world, reward, reg_reward, discount)

    if draw: utils.plot_rewards(world, reward, expert_policy, trajectories, phi, theta_regular)
    # update theta:
    if debug: print("theta")
    theta = update_theta(theta, phi, meta_lr, phi_loss, debug)
    if debug: print("phi")
    phi = update_theta(None, phi, meta_lr, phi_loss, False)
    if debug: print("theta - regular")
    theta_regular = update_theta(None, theta_regular, meta_lr, reg_loss, False)

    if debug: print("phi loss: {0}  :   regular loss: {1}".format(phi_loss, reg_loss))

    return theta, phi, theta_regular, reward, world, phi_loss, reg_loss


def update_theta(theta, phi, meta_lr, loss, debug):
    """
    Update theta
    """

    # normalize phi
    phi = phi / phi.max()
    if theta is None: theta = phi #/ phi.shape[0]

    phi_mat = phi.reshape(int(np.sqrt(phi.shape[0])), -1)
    gauss_phi = Gfit.fitgaussian(phi_mat)
    phi_fit = Gfit.gaussGrid(phi_mat.shape, *gauss_phi)

    theta_mat = theta.reshape(int(np.sqrt(theta.shape[0])), -1)
    gauss_theta = Gfit.fitgaussian(theta_mat)

    # theta = theta + meta_lr * (phi - theta)
    gauss_theta = gauss_theta + loss * meta_lr * (gauss_phi - gauss_theta)
    theta_mat = Gfit.gaussGrid(phi_mat.shape, *gauss_theta)
    theta = theta_mat.reshape(-1)
    # normalize theta:
    theta = theta / theta.max()

    if debug: print(loss * meta_lr * (gauss_phi - gauss_theta))
    return theta


def validate(size, world, optimal_policy_value, agent_policy_value):
    agent_policy = np.array([
        np.argmax([agent_policy_value[world.state_index_transition(s, a)] for a in range(world.n_actions)])
        for s in range(world.n_states)
    ])

    optimal_options = []
    for s in range(world.n_states):
        values = [optimal_policy_value[world.state_index_transition(s, a)] for a in range(world.n_actions)]
        optimal_options.append(np.argwhere(values == np.amax(values)))

    # compare the policies, remember that the terminal state's policy is unneeded
    error_num = sum([agent_policy[s] not in optimal_options[s] for s in range(world.n_states)])
    return error_num / size**2


def calc_rewards(world, gt_reward, maml_reward, reg_reward, size, discount, debug=False):
     # optimal policy:
    optimal_policy_value = Solver.optimal_policy_value(world, gt_reward, discount)
    maxent_policy_value = Solver.optimal_policy_value(world, maml_reward, discount)
    reg_maxent_policy_value = Solver.optimal_policy_value(world, reg_reward, discount)

    # validate
    policy_score = validate(size, world, optimal_policy_value, maxent_policy_value)
    reg_policy_score = validate(size, world, optimal_policy_value, reg_maxent_policy_value)
    validation_score = sum((maml_reward - gt_reward)**2)
    regular_score = sum((reg_reward - gt_reward)**2)
    return validation_score, regular_score, policy_score, reg_policy_score


def maml_step(
    data,
    theta,
    batch_size,
    meta_lr,
    size,
    p_slip,
    terminal,
    debug,
    discount,
    draw
):
    startTime = time.time()
    theta, phi, theta_regular, gt_reward, world, phi_loss, reg_loss = maml_iteration(
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

    # rewards:
    features = World.state_features(world)
    mamlReward = features.dot(phi)
    regularReward = features.dot(theta_regular)

    validation_score, regular_score, policy_score, reg_policy_score = calc_rewards(
        world,
        gt_reward,
        mamlReward,
        regularReward,
        size,
        discount,
        debug
    )

    data["thetas"].append(theta.copy())
    data["groundTruthReward"].append(gt_reward)
    data["phi_loss"].append(phi_loss)
    data["reg_loss"].append(reg_loss)
    data["mamlReward"].append(mamlReward)
    data["regularReward"].append(regularReward)
    data["worlds"].append(world)
    data["validation_score"].append(validation_score)
    data["regular_score"].append(regular_score)
    data["policy_score"].append(policy_score)
    data["reg_policy_score"].append(reg_policy_score)

    executionTime = (time.time() - startTime)
    if debug:
            print('Execution time: {0} (sec) - \
                policy score: {1}, regular policy score: {2}'.
                format(
                    round(executionTime, 2),
                    policy_score,
                    reg_policy_score
                )
            )

    return theta

def maml(
    N=100,
    batch_size=20,
    meta_lr=0.1,
    size=5,
    p_slip=0,
    terminal=None,
    debug=False,
    theta=None,
    discount=0.7,
    draw=False,
    validateStep=100000
):
    data = {
        "thetas": [],
        "groundTruthReward": [],
        "phi_loss": [],
        "reg_loss": [],
        "mamlReward": [],
        "regularReward": [],
        "worlds": [],
         "validation_score": [],
         "regular_score": [],
         "policy_score": [],
         "reg_policy_score": []
    }

    valid_data = data.copy()

    for ind in range(N):
        if debug: print("Iteration #{0}".format(ind))
        theta = maml_step(data, theta, batch_size, meta_lr, size, p_slip, terminal, debug, discount, draw)
        if ind is validateStep:
            print("Validation for step #{0}".format(ind))
            _ = maml_step(valid_data, theta, batch_size, meta_lr, size, p_slip, terminal, True, discount, draw)


    return data, valid_data

#%%
if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 5
    p_slip = 0.5
    N = 200
    validateStep = 2
    batch_size = 10
    meta_lr = 0.1
    terminal = None
    debug = False
    data, valid_data = maml(
        N=N,
        batch_size=batch_size,
        meta_lr=meta_lr,
        size=size,
        p_slip=p_slip,
        terminal=terminal,
        debug=debug,
        draw=False,
        validateStep=validateStep
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
    fig = plt.figure(figsize=(12,8))
    plt.plot(range(N), data["phi_loss"][:N], data["reg_loss"][:N])
    plt.legend(["phi_loss", "reg_loss"])
    plt.title("Loss for mandril, vs. loss for regular maxent for p_slip of: {0}".format(p_slip))
    plt.show()