# state-space-abstraction-hierarchical-rl

In Hierarchical Reinforcement Learning, as described by Kulkarni et al. (https://arxiv.org/abs/1604.06057), a reinforcement learning agent is split into two components: a meta-controller, which selects subgoals, and a controller, which learns to complete those subgoals. The meta-controller learns to pick a sequence of subgoals that optimally completes the given task, while the controller learns which sequence of primitive actions it should take to complete each subgoal. This approach performs a temporal abstraction of a reinforcement learning agent's actions, and it addresses the problems of exploration and reward sparsity.

In this exploratory project, we tried to incorporate state space abstraction into this framework. In Kulkarni et al., both the meta-controller and controller are implemented as DQNs, and they both receive the same environment state, while the controller also receives the subgoal selected by the meta-controller. This state space is often continuous or very large, such as the set of all possible game images containing pixel values, hence using a DQN rather than tabular Q-learning is necessary.

If we are able to discretize the meta-controller's state space and represent the subgoals in that new state space, then the meta-controller can be trained using tabular Q-learning. In theory, this would significantly speed up the overall training of the reinforcement learning agent. 

To illustrate how this would work, consider the [MountainCar](https://github.com/openai/gym/wiki/MountainCar-v0) environment. In this environment, the agent controls a car, which can move left or right, and the goal is to reach the top of the hill on the right side. The agent's state is a vector of two numbers: the current position and velocity.

![alt text](https://cdn-images-1.medium.com/max/1600/1*nbCSvWmyS_BUDz_WAJyKUw.gif)

In order to discretize this state space, we collect 100 trajectories of the agent by spawning the agent in a random location on the hill in every episode and picking random actions until the episode terminates. These trajectories give us thousands of states which we then cluster using k-means clustering on the position and velocity. This clustering process is almost entirely unsupervised with the caveat that we cluster goal states into a separate cluster. Below is an example of the formed clusters when k=5.

![alt_text](https://github.com/skumar9876/state-space-abstraction-hierarchical-rl/blob/master/clusters/Clusters.png)

Now, the meta-controller's state is a one-hot vector indicating in which cluster the agent is currently located. Its action space is to instruct the agent to go to one of the 5 clusters. Note that the meta-controller receives high negative reward if it instructs the agent to go to a cluster it is already in so that does not learn to do this, as this is a suboptimal strategy the meta-controller would otherwise learn. 

The controller's state is the environment state, which is the 1-dimensional vector of the position and velocity, concatenated with the one-hot vector representing the meta-controller's instruction. Thus, the meta-controller is a tabular Q-learning agent, while the controller is a DQN agent. We wanted to see if this setup would reduce the training time of the agent when compared to a standard DQN agent.

Below is a plot summarizing the results:


Other variants we tried (none of them beat the standard DQN agent):
- Increasing the number of clusters to boost the granularity of the meta-controller state space (this also increases the action space of the meta-controller).
- Adding a memory to the meta-controller and implementing it as a DQN with an RNN generating the q-values. This setup did allow the meta-controller to learn to move the agent back and forth before instructing it to go to the "goal cluster," but it was very unstable.
