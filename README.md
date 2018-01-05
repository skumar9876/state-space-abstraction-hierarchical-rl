# state-space-abstraction-hierarchical-rl

In Hierarchical Reinforcement Learning, as described by Kulkarni et al. (https://arxiv.org/abs/1604.06057), a reinforcement learning agent is split into two components: a meta-controller, which selects subgoals, and a controller, which learns to complete those subgoals. The meta-controller learns to pick a sequence of subgoals that optimally completes the given task, while the controller learns which sequence of primitive actions it should take to complete each subgoal. This approach performs a temporal abstraction of a reinforcement learning agent's actions, and it addresses the problems of exploration and reward sparsity.

In this exploratory project, we tried to incorporate state space abstraction into this framework. In Kulkarni et al., both the meta-controller and controller are implemented as DQNs, and they both receive the same environment state, while the controller also receives the subgoal selected by the meta-controller. This state space is often continuous or very large, such as the set of all possible game images containing pixel values, hence using a DQN rather than tabular Q-learning is necessary.

If we are able to discretize the meta-controller's state space and represent the subgoals in that new state space, then the meta-controller can be trained using tabular Q-learning. In theory, this would significantly speed up the overall training of the reinforcement learning agent. 

To illustrate how this would work, consider the [MountainCar](https://github.com/openai/gym/wiki/MountainCar-v0) environment. In this environment, the agent controls a car, which can move left or right, and the goal is to reach the top of the hill on the right side. The agent's state is a vector of two numbers: the current position and velocity.

![alt text](https://cdn-images-1.medium.com/max/1600/1*nbCSvWmyS_BUDz_WAJyKUw.gif)

In order to discretize this state space, we collect 100 trajectories of the agent by spawning the agent in a random location on the hill in every episode and picking random actions until the episode terminates. These trajectories give us thousands of states which we then cluster using k-means clustering on the position and velocity. This clustering process is almost entirely unsupervised with the caveat that we cluster goal states into a separate cluster. Below is an example of the formed clusters when k=5.

![alt_text](https://github.com/skumar9876/state-space-abstraction-hierarchical-rl/blob/master/clusters/Clusters.png)

Now, the meta-controller's state is a one-hot vector indicating in which cluster the agent is currently located. Its action space is to instruct the agent to go to one of the 5 clusters. Note that we provide the meta-controller with high negative reward if it instructs the agent to go to a cluster it is already in so that the meta-controler does not learn to do this, as this is an easily found sub-optimal strategy with respect to the meta-controller's received reward. 

The controller's state is the environment state, which is the 1-dimensional vector of the position and velocity, concatenated with the one-hot vector representing the meta-controller's instruction. Thus, the meta-controller is a tabular Q-learning agent, while the controller is a DQN agent. We wanted to see if this setup would reduce the training time of the agent when compared to a standard DQN agent.

Below is a plot summarizing the results:
![alt_text](https://github.com/skumar9876/state-space-abstraction-hierarchical-rl/blob/master/results/plot.png)

We ran 10 experiments with DQN and 10 with this hierarchical reinforcement learning setup. The plot shows the range between the 10th and 90th percentile of episodic rewards achieved over training time. The x-axis is number of training steps, while the y-axis is the reward. The red is standard DQN while the green is our approach. It is evident from the plot that our setup does not produce a clear benefit over the standard DQN approach. Upon analyzing the meta-controller's behavior, we found the following problems:
- The clusters are not granular enough. There is too wide of a range in position / velocity within each cluster. This means that there is a large variety of environment states that the agent can be in within each cluster, making a single policy for each cluster insufficient for capturing the optimal behavior required for this environment.
- Each state can only have one possible greedy action with tabular Q learning. This does not account for the fact that the agent has to move back and forth a couple times before gaining enough momentum to travel up the hill. The meta-controller first learns to move the agent back and forth but then cannot instruct the agent to reach the goal state. Eventually, the meta-controller converges on the policy of immediately telling the agent to go to the goal state, rendering the use of hierarchy ineffective, as the controller now has to solve the same problem as a standard DQN agent. 

Other variants we tried (none of them beat the standard DQN agent):
- Increasing the number of clusters to boost the granularity of the meta-controller state space (this also increases the action space of the meta-controller).
- Adding a memory to the meta-controller and implementing it as a DQN with an RNN generating the q-values. This setup did allow the meta-controller to learn to move the agent back and forth before instructing it to go to the "goal cluster," but the training was very unstable.

The goal of this approach was to speed up deep reinforcement learning agents' training by combining unsupervised state space clustering with the hierarchical reinforcement learning paradigm. Perhaps if we select an environment such that within each cluster, all the environment states are the "same" with respect to the optimal policy (e.g. in an environment with rooms, if the agent is in one room, its position in the room may not affect which room it should go to next to solve the task), our approach would show benefits over the baseline.
