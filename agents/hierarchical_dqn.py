"""
Hierarchical DQN implementation as described in Kulkarni et al.
https://arxiv.org/pdf/1604.06057.pdf
@author: Saurabh Kumar
"""

from collections import defaultdict
from controller_dqn import ControllerDqnAgent
from dqn import DqnAgent
from lstm_dqn import LstmDqnAgent
import numpy as np
from qLearning import QLearningAgent
import sys


class HierarchicalDqnAgent(object):
    INTRINSIC_STEP_COST = -1    # Step cost for the controller.

    INTRINSIC_TIME_OUT = 50             # Number of steps after which intrinsic episode ends.
    INTRINSIC_TIME_OUT_PENALTY = -10    # Penalty given to controller for timing out episode.

    ARTIFICIAL_PENALTY = -100   # Penalty given to the meta-controller for telling the
                                # agent to go to the same cluster it is already in.
    EXTRA_TRAVEL_PENALTY = -1   # Penalty given to meta-controller if controller agent
                                # travels through additional clusters to get to target cluster.
    PRETRAIN_EPISODES = 100

    def __init__(self,
                 learning_rates=[0.1, 0.00025],
                 state_sizes=[0, 0],
                 agent_types=['network', 'network'],
                 subgoals=None,
                 num_subgoals=0,
                 num_primitive_actions=0,
                 meta_controller_state_fn=None,
                 check_subgoal_fn=None,
                 use_extra_travel_penalty=False,
                 use_extra_bit_for_subgoal_center=False,
                 use_controller_dqn=False,
                 use_intrinsic_timeout=False,
                 use_memory=False,
                 memory_size=0,
                 pretrain_controller=False):
        print "h-DQN"
        print "Use extra travel penalty:"
        print use_extra_travel_penalty
        print "Use extra bit for subgoal center:"
        print use_extra_bit_for_subgoal_center
        print "Use controller dqn:"
        print use_controller_dqn
        print "Use intrinsic timeout:"
        print use_intrinsic_timeout
        print "Use memory:"
        print use_memory
        print "Memory size:"
        print memory_size
        print "Pretrain Controller:"
        print pretrain_controller
        """Initializes a hierarchical DQN agent.

           Args:
            learning_rates: learning rates of the meta-controller and controller agents.
            state_sizes: state sizes of the meta-controller and controller agents.
            agent_types: type of each agent - either tabular QLearning agent or Deep Q Network.
            subgoals: array of subgoals for the meta-controller.
            num_subgoals: the action space of the meta-controller.
            num_primitive_actions: the action space of the controller.
            meta_controller_state_fn: function that returns the state of the meta-controller.
            check_subgoal_fn: function that checks if agent has satisfied a particular subgoal.
            use_extra_travel_penalty: whether or not to penalize the meta-controller for bad instructions.
            use_extra_bit_for_subgoal_center: whether or not to use an extra bit to indicate whether
                                              agent is at center of a particular cluster.
            use_controller_dqn: whether to use regular dqn or controller dqn for the controller.
            use_intrinsic_timeout: whether or not to intrinsically timeout the controller.
        """
        if not use_extra_travel_penalty:
            self.EXTRA_TRAVEL_PENALTY = 0

        if use_extra_bit_for_subgoal_center:
            self.ARTIFICIAL_PENALTY = 0
            state_sizes[0] = state_sizes[0] * 2

        if not pretrain_controller:
            self.PRETRAIN_EPISODES = 0

        if use_memory:
            print "Decaying meta-controller epsilon faster!"
            self._meta_controller = LstmDqnAgent(num_actions=num_subgoals,
                                                 state_dims=[memory_size],
                                                 sequence_length=memory_size,
                                                 replay_memory_init_size=100,
                                                 target_update=100,
                                                 epsilon_end=0.01,
                                                 epsilon_decay_steps=5000)
        else:
            self._meta_controller = QLearningAgent(num_states=state_sizes[0],
                                                   num_actions=num_subgoals,
                                                   learning_rate=learning_rates[0],
                                                   epsilon=0.1)
        if use_controller_dqn:
            self._controller = ControllerDqnAgent(learning_rate=learning_rates[1],
                num_actions=num_primitive_actions,
                state_dims=state_sizes[1],
                subgoal_dims=[num_subgoals])
        else:
            print "Epsilon end for controller is 0.01!"
            self._controller = DqnAgent(learning_rate=learning_rates[1],
                num_actions=num_primitive_actions,
                state_dims=[state_sizes[1][0] + num_subgoals],
                epsilon_end=0.01) # CHANGED

        self._subgoals = subgoals
        self._num_subgoals = num_subgoals

        self._meta_controller_state_fn = meta_controller_state_fn
        self._check_subgoal_fn = check_subgoal_fn

        self._use_extra_bit_for_subgoal_center = use_extra_bit_for_subgoal_center
        self._use_controller_dqn = use_controller_dqn

        self._use_intrinsic_timeout = use_intrinsic_timeout

        self._use_memory = use_memory
        self._memory_size = memory_size


        self._meta_controller_state = None
        self._curr_subgoal = None
        self._meta_controller_reward = 0
        self._intermediate_clusters = []
        self._intermediate_dict = defaultdict(int)
        self._intermediate_clusters_dict = defaultdict(int)
        self._history = [0 for i in xrange(self._memory_size)]

        # Only used if use_extra_bit_for_subgoal_center is True.
        self._original_state = None

        self._next_meta_controller_state = None

        self._intrinsic_time_step = 0

        self._episode = 0

    def update_history(self, state):
        returned_state = state
        if self._meta_controller_state_fn:
            returned_state = self._meta_controller_state_fn(state, self._original_state)

        current_cluster_id = np.where(np.squeeze(returned_state) == 1)[0][0] + 1
        new_history = self._history[1:]

        # print "History update!"
        # print self._history
        # print new_history
        # print current_cluster_id
        new_history.append(current_cluster_id)
        # print new_history
        # print ""
        self._history = new_history

    def get_meta_controller_state(self, state):
        returned_state = state
        if self._meta_controller_state_fn:
            returned_state = self._meta_controller_state_fn(state, self._original_state)

        if self._use_memory:
            returned_state = self._history[:]

        return returned_state

    def get_controller_state(self, state, subgoal_index):
        curr_subgoal = self._subgoals[subgoal_index]

        # Concatenate the environment state with the subgoal.
        controller_state = list(state[0])
        for i in xrange(len(curr_subgoal)):
            controller_state.append(curr_subgoal[i])
        controller_state = np.array([controller_state])
        # print controller_state
        return np.copy(controller_state)

    def intrinsic_reward(self, state, subgoal_index):
        if self._use_intrinsic_timeout and self._intrinsic_time_step >= self.INTRINSIC_TIME_OUT:
            return self.INTRINSIC_TIME_OUT_PENALTY
        if self.subgoal_completed(state, subgoal_index):
            return 1
        else:
            return self.INTRINSIC_STEP_COST

    def subgoal_completed(self, state, subgoal_index):
        if self._check_subgoal_fn is None:
            if self._use_intrinsic_timeout and self._intrinsic_time_step >= self.INTRINSIC_TIME_OUT:
                return True
            return state == self._subgoals[subgoal_index]
        else:
            if self._use_intrinsic_timeout and self._intrinsic_time_step >= self.INTRINSIC_TIME_OUT:
                return True

            if not self._use_memory and self._meta_controller_state[self._curr_subgoal] == 1:
                if np.sum(self._meta_controller_state) > 1:
                    return False

                return self._check_subgoal_fn(state, subgoal_index, self._original_state)
            else:
                return self._check_subgoal_fn(state, subgoal_index)

    def store(self, state, action, reward, next_state, terminal, eval=False):
        """Stores the current transition in replay memory.
           The transition is stored in the replay memory of the controller.
           If the transition culminates in a subgoal's completion or a terminal state, a
           transition for the meta-controller is constructed and stored in its replay buffer.

           Args:
            state: current state
            action: primitive action taken
            reward: reward received from state-action pair
            next_state: next state
            terminal: extrinsic terminal (True or False)
            eval: Whether the current episode is a train or eval episode.
        """

        self._meta_controller_reward += reward
        self._intrinsic_time_step += 1

        # Compute the controller state, reward, next state, and terminal.
        intrinsic_state = self.get_controller_state(state, self._curr_subgoal)
        intrinsic_next_state = self.get_controller_state(next_state, self._curr_subgoal)
        intrinsic_reward = self.intrinsic_reward(next_state, self._curr_subgoal)
        subgoal_completed = self.subgoal_completed(next_state, self._curr_subgoal)
        intrinsic_terminal = subgoal_completed or terminal

        self._controller.store(np.copy(intrinsic_state), action,
            intrinsic_reward, np.copy(intrinsic_next_state), intrinsic_terminal, eval)

        # Check for intermediate state.
        intermediate_meta_controller_state = self.get_meta_controller_state(next_state)

        if not self._use_memory:
            intermediate_cluster_id = np.where(np.squeeze(intermediate_meta_controller_state) == 1)[0][0]
        else:
            intermediate_cluster_id = intermediate_meta_controller_state[-1] - 1

        self._intermediate_dict[intermediate_cluster_id] += 1
        # Agent is traveling through a cluster that is not the starting or ending cluster.
        # FIX THIS!!!!
        if list(intermediate_meta_controller_state[0:self._num_subgoals]) != list(
            self._meta_controller_state[0:self._num_subgoals]) and not subgoal_completed:
            self._meta_controller_reward += self.EXTRA_TRAVEL_PENALTY


            self._intermediate_clusters.append(intermediate_cluster_id)
            self._intermediate_clusters_dict[intermediate_cluster_id] += 1

        if terminal and not eval:
            self._episode += 1

        if subgoal_completed or terminal:
            # Normalize the meta-controller reward.
            self._meta_controller_reward /= 100.0

            meta_controller_state = np.copy(self._meta_controller_state)
            if not self._use_memory:
                next_meta_controller_state = self.get_meta_controller_state(next_state)
            else:
                returned_state = self._meta_controller_state_fn(next_state, self._original_state)
                current_cluster_id = np.where(np.squeeze(returned_state) == 1)[0][0] + 1
                new_history = self._history[1:]
                new_history.append(current_cluster_id)
                next_meta_controller_state = new_history

            if self._episode >= self.PRETRAIN_EPISODES:
                self._meta_controller.store(np.copy(meta_controller_state), self._curr_subgoal,
                    self._meta_controller_reward, np.copy(next_meta_controller_state),
                    terminal, eval, reward)

            if eval:
                if subgoal_completed:
                    print "Subgoal completed!"
                    print "Intermediate Clusters:"
                    print self._intermediate_clusters
                    print "Intermediate Cluster Count:"
                    print self._intermediate_dict
                    print "Intermediate non-beginning cluster count:"
                    print self._intermediate_clusters_dict
                    print "State:"
                    print next_state
                    print "Meta-Controller reward:"
                    print self._meta_controller_reward
                    print "Intrinsic reward:"
                    print intrinsic_reward
                    print "Cluster:"
                    print next_meta_controller_state
                    print ""
                    print ""
                else:
                    print "Terminal!"
                    print "Intermediate clusters:"
                    print self._intermediate_clusters
                    print "Intermediate cluster count:"
                    print self._intermediate_dict
                    print "Intermediate non-beginning cluster count:"
                    print self._intermediate_clusters_dict
                    print "State:"
                    print next_state
                    print "Meta-Controller reward:"
                    print self._meta_controller_reward
                    print "Intrinsic reward:"
                    print intrinsic_reward
                    print "Cluster:"
                    print next_meta_controller_state
                    print ""
                    print ""

            # Reset the current meta-controller state and current subgoal to be None
            # since the current subgoal is finished. Also reset the meta-controller's reward.
            self._next_meta_controller_state = np.copy(next_meta_controller_state)

            if terminal:
                self._next_meta_controller_state = None

            self._meta_controller_state = None
            self._curr_subgoal = None
            self._meta_controller_reward = 0

            self._intermediate_clusters = []
            self._intermediate_dict = defaultdict(int)
            self._intermediate_clusters_dict = defaultdict(int)

            self._original_state = None
            self._intrinsic_time_step = 0

            if terminal:
                self._history = [0 for i in xrange(self._memory_size)]

    def sample(self, state):
        """Samples an action from the hierarchical DQN agent.
           Samples a subgoal if necessary from the meta-controller and samples a primitive action
           from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: a primitive action.
        """
        if self._meta_controller_state is None:
            if self._use_memory:
                self.update_history(state)

            if self._next_meta_controller_state is not None and not self._use_memory:
                self._meta_controller_state = self._next_meta_controller_state
            else:
                self._meta_controller_state = self.get_meta_controller_state(state)

            self._curr_subgoal = self._meta_controller.sample([self._meta_controller_state])

            # Artificially penalize the meta-controller for picking the subgoal to
            # be the same as the current cluster.
            if self._use_memory:
                same_cluster_instruction = (self._meta_controller_state[-1] - 1) == self._curr_subgoal
            else:
                same_cluster_instruction = self._meta_controller_state[self._curr_subgoal] == 1

            if same_cluster_instruction:
                self._meta_controller_reward = self.ARTIFICIAL_PENALTY
                self._original_state = state

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.sample(controller_state)

        return action

    def best_action(self, state):
        """Returns the greedy action from the hierarchical DQN agent.
           Gets the greedy subgoal if necessary from the meta-controller and gets
           the greedy primitive action from the controller.

           Args:
            state: the current environment state.

           Returns:
            action: the controller's greedy primitive action.
        """
        returned_info = None

        if self._meta_controller_state is None:
            if self._use_memory:
                self.update_history(state)

            if self._next_meta_controller_state is not None and not self._use_memory:
                self._meta_controller_state = self._next_meta_controller_state
            else:
                self._meta_controller_state = self.get_meta_controller_state(state)

            self._curr_subgoal = self._meta_controller.best_action([self._meta_controller_state])

            returned_info = [self._meta_controller_state, self._curr_subgoal]

            # Artificially penalize the meta-controller for picking the subgoal to
            # be the same as the current cluster.
            if self._use_memory:
                same_cluster_instruction = (self._meta_controller_state[-1] - 1) == self._curr_subgoal
            else:
                same_cluster_instruction = self._meta_controller_state[self._curr_subgoal] == 1

            if same_cluster_instruction:
                self._meta_controller_reward = self.ARTIFICIAL_PENALTY
                self._original_state = state

            print "Current State:"
            print state
            print "Current Meta-Controller State:"
            print self._meta_controller_state
            print "Current subgoal picked:"
            print self._curr_subgoal

        controller_state = self.get_controller_state(state, self._curr_subgoal)
        action = self._controller.best_action(controller_state)
        return action, returned_info

    def update(self):
        self._controller.update()
        # Only update meta-controller right after a meta-controller transition has taken place,
        # which occurs only when either a subgoal has been completed or the agnent has reached a
        # terminal state.
        if self._meta_controller_state is None:
            self._meta_controller.update()
