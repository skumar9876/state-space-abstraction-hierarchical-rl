# state-space-abstraction-hierarchical-rl

In Hierarchical Reinforcement Learning, as described by Kulkarni et al. (https://arxiv.org/abs/1604.06057), a reinforcement learning agent is split into two components: a meta-controller, which selects subgoals, and a controller, which learns to complete those subgoals. The meta-controller learns to pick a sequence of subgoals that optimally completes the given task, while the controller learns which sequence of primitive actions it should take to complete each subgoal. This approach performs a temporal abstraction of a reinforcement learning agent's actions, and it addresses the problems of exploration and reward sparsity.

In this exploratory project, I tried to incorporate state space abstraction into this framework. In Kulkarni et al., both the meta-controller and controller are implemented as DQNs, and they both receive the same environment state, while the controller also receives the subgoal selected by the meta-controller. This state space is often continuous, such as a game image containing pixel values, hence using a DQN rather than tabular Q-learning is necessary.

If we are able to discretize the meta-controller's state space represent the subgoals with respect to that new state space, then the meta-controller can be trained using tabular Q-learning. In theory, this would significantly speed up the overall training of the reinforcement learning agent. 

suppose we abstract or decompose the meta-controller's state space so that it becomes discrete and 
