#!/usr/bin/env python

import utils
import solver as Solver
import gridworld as World
import gaussianfit as Gfit

import numpy as np
import matplotlib.pyplot as plt

import time

class Mandril:
    def __init__(
        self,
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
        validate_step=100000,
        model="Gaussian"
    ):
        self.N = N
        self.batch_size = batch_size
        self.meta_lr = meta_lr
        self.size = size
        self.p_slip = p_slip
        self.terminal = terminal
        self.debug = debug
        self.theta = theta
        self.discount = discount
        self.draw = draw
        self.validate_step = validate_step
        self.model = model


    def maml(self, theta=None):
        if theta is None: theta = self.theta

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
        valid_data = {
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

        for ind in range(self.N):
            if self.debug: print("Iteration #{0}".format(ind))
            theta = self.maml_step(data, theta)
            if ind % self.validate_step == 0:
                print("Validation for step #{0}".format(ind))
                real_debug = self.debug
                self.debug = True
                _ = self.maml_step(valid_data, theta)
                self.debug = real_debug

        return data, valid_data


    def maml_step(self, data, theta):
        startTime = time.time()
        theta, phi, theta_regular, gt_reward, world, phi_loss, reg_loss = self.maml_iteration(theta)

        # rewards:
        features = World.state_features(world)
        mamlReward = features.dot(phi)
        regularReward = features.dot(theta_regular)

        validation_score, regular_score, policy_score, reg_policy_score = self.calc_rewards(
            world,
            gt_reward,
            mamlReward,
            regularReward
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
        if self.debug:
                print('Execution time: {0} (sec) - \
                    policy score: {1}, regular policy score: {2}'.
                    format(
                        round(executionTime, 2),
                        policy_score,
                        reg_policy_score
                    )
                )

        return theta


    def maml_iteration(self, theta):
        # set-up mdp
        world, reward, terminal = utils.setup_mdp(self.size, self.p_slip, location=self.terminal)
        # get expert trajectories
        trajectories, expert_policy = utils.generate_trajectories(
            world,
            reward,
            terminal,
            n_trajectories=self.batch_size,
            discount=self.discount
        )

        # optimize with maxent
        phi, phi_reward = utils.maxent(
            world,
            terminal,
            trajectories,
            theta
        )
        phi_loss = self.get_loss(world, reward, phi_reward)

        # Get a theta for an untrained init:
        theta_regular, reg_reward = utils.maxent(
            world,
            terminal,
            trajectories
        )
        reg_loss = self.get_loss(world, reward, reg_reward)

        if self.draw: utils.plot_rewards(world, reward, expert_policy, trajectories, phi, theta_regular)
        # update theta:
        theta = self.update_theta(theta, phi, self.meta_lr, phi_loss)
        phi = self.update_theta(None, phi, self.meta_lr, phi_loss)
        theta_regular = self.update_theta(None, theta_regular, self.meta_lr, reg_loss)

        if self.debug: print("phi loss: {0}  :   regular loss: {1}".format(phi_loss, reg_loss))

        return theta, phi, theta_regular, reward, world, phi_loss, reg_loss


    def get_loss(self, world, gt_reward, reward):
        # Calculate loss:
        optimal_policy_value = Solver.optimal_policy_value(world, gt_reward, self.discount)
        maxent_policy_value = Solver.optimal_policy_value(world, reward, self.discount)

        # validate
        loss = self.validate(world, optimal_policy_value, maxent_policy_value)
        return loss


    def update_theta(self, theta, phi, meta_lr, loss):
        """
        Update theta
        """

        # normalize phi
        phi = phi / phi.max()
        if theta is None: theta = phi #/ phi.shape[0]

        if self.model == "Gaussian":
            phi_mat = phi.reshape(int(np.sqrt(phi.shape[0])), -1)
            gauss_phi = Gfit.fitgaussian(phi_mat)
            # phi_fit = Gfit.gaussGrid(phi_mat.shape, *gauss_phi)

            theta_mat = theta.reshape(int(np.sqrt(theta.shape[0])), -1)
            gauss_theta = Gfit.fitgaussian(theta_mat)

            # theta = theta + meta_lr * (phi - theta)
            gauss_theta = gauss_theta + loss * meta_lr * (gauss_phi - gauss_theta)
            theta_mat = Gfit.gaussGrid(phi_mat.shape, *gauss_theta)
            theta = theta_mat.reshape(-1)
            # normalize theta:
            theta = theta / theta.max()
        elif self.model == "Naive":
            theta = theta + loss * meta_lr * (phi - theta)
        else:
            raise ValueError("model is undefined")
        return theta


    def validate(self, world, optimal_policy_value, agent_policy_value):
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
        return error_num / self.size**2


    def calc_rewards(self, world, gt_reward, maml_reward, reg_reward):
        # optimal policy:
        optimal_policy_value = Solver.optimal_policy_value(world, gt_reward, self.discount)
        maxent_policy_value = Solver.optimal_policy_value(world, maml_reward, self.discount)
        reg_maxent_policy_value = Solver.optimal_policy_value(world, reg_reward, self.discount)

        # validate
        policy_score = self.validate(world, optimal_policy_value, maxent_policy_value)
        reg_policy_score = self.validate(world, optimal_policy_value, reg_maxent_policy_value)
        validation_score = sum((maml_reward - gt_reward)**2)
        regular_score = sum((reg_reward - gt_reward)**2)
        return validation_score, regular_score, policy_score, reg_policy_score


#%% MAIN

if __name__ == '__main__':
    startTime = time.time()
    # parameters
    size = 5
    p_slip = 0.5
    N = 10
    validate_step = 2
    batch_size = 10
    meta_lr = 0.1
    terminal = None
    debug = False

    model = "Naive"

    # Mandril class:
    mandril = Mandril(
        N=N,
        batch_size=batch_size,
        meta_lr=meta_lr,
        size=size,
        p_slip=p_slip,
        terminal=terminal,
        debug=debug,
        validate_step=validate_step,
        model=model
    )
    # run maml:
    data, valid_data = mandril.maml()

    # Print output:
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