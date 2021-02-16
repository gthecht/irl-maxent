import numpy as np
import gridworld as World

class Agent:
  def __init__(self, size=None, max_steps=100, theta=None):
    self.theta = theta
    self.max_steps = max_steps
    # this world is a deterministic one of size
    self.world = World.GridWorld(size=size)

  def get_policy(self, theta=None, size=None):
    # If no theta was given use the self theta,
    # maybe update self.theta if it is given?
    if theta is not None: self.theta = theta
    if size is not None: self.world = World.GridWorld(size=size)
    policy = np.zeros(self.world.n_states)
    for state in range(self.world.n_states):
      next_state = np.array([self.world.state_index_transition(state, action)
                    for action in range(self.world.n_actions)])
      next_potentials = theta[next_state]
      next_potentials[next_state == state] = -np.inf
      policy[state] = next_potentials.argmax()
    return policy

  def solve(self, start, terminal, theta=None, size=None):
    """
    Solve the given problem with a given theta
    """
    # If no theta was given use the self theta,
    # maybe update self.theta if it is given?
    if theta is not None: self.theta = theta
    if size is not None: self.world = World.GridWorld(size=size)

    # iterate until arrived at terminal, or taken max_steps
    step_num = 0
    current_state = start
    actions = []
    states = []
    while step_num < self.max_steps:
      action, current_state = self.step(current_state)
      actions.append(action)
      states.append(current_state)
      if current_state in terminal: break
    return actions, states

  def step(self, current_state):
    """
    Take a step from the current state to the adjacent square with the greatest
    value

    Args:
      current_state: The current state

    Returns:
      The next state
    """

    # Get the states and potentials for the possible actions - note that if an
    # action keeps the agent in the same place, we won't allow it since this
    # will continue eternally.
    potentials = np.zeros(self.world.n_actions, 1)
    for action in self.world.n_actions:
      next_s = self.world.state_index_transition(current_state, action)
      if next_s is current_state: potentials[action] = -np.inf
    action = potentials.argmax()
    next_state = self.world.state_index_transition(current_state, action)
    return action, next_state