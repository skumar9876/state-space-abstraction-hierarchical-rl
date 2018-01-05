from dqn import DqnAgent
import tensorflow as tf
from dqn import clipped_error

class ControllerDqnAgent(DqnAgent):


    def __init__(self, subgoal_dims=[], *args, **kwargs):
        self._subgoal_dims = subgoal_dims
        super(ControllerDqnAgent, self).__init__(*args, **kwargs)

    def _q_network(self, state, subgoal):
        state_layer1 = tf.contrib.layers.fully_connected(state, 64, activation_fn=tf.nn.relu)
        subgoal_layer1 = tf.contrib.layers.fully_connected(subgoal, 64, activation_fn=tf.nn.relu)

        layer1 = tf.concat([state_layer1, subgoal_layer1], axis=1)

        q_values = tf.contrib.layers.fully_connected(layer1, self._num_actions, activation_fn=None)

        return q_values

    def _construct_graph(self):
        # state_shape=[None]
        # subgoal_shape=[None]
        # for dim in self._state_dims:
        #    state_shape.append(dim)
        # for dim in self._subgoal_dims:
        #    subgoal_shape.append(dim)
        state_shape = self._state_dims[0]
        subgoal_shape = self._subgoal_dims[0]

        self._state = tf.placeholder(shape=[None, state_shape + subgoal_shape],
            dtype=tf.float32)
        self._controller_state, self._subgoal = tf.split(
            self._state, [state_shape, subgoal_shape], axis=1)

        with tf.variable_scope('q_network'):
            self._q_values = self._q_network(self._controller_state, self._subgoal)
        with tf.variable_scope('target_q_network'):
            self._target_q_values = self._q_network(self._controller_state, self._subgoal)
        with tf.variable_scope('q_network_update'):
            self._picked_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32)
            self._td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

            self._q_values_pred = tf.gather_nd(self._q_values, self._picked_actions)

            self._losses = clipped_error(self._q_values_pred - self._td_targets)
            self._loss = tf.reduce_mean(self._losses)

            self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate, 0.99, 0.0, 1e-6)
            # self.optimizer = tf.train.AdamOptimizer(0.0001)
            # self.optimizer = tf.train.GradientDescentOptimizer(0.1)

            grads_and_vars = self.optimizer.compute_gradients(self._loss, tf.trainable_variables())

            grads = [gv[0] for gv in grads_and_vars]
            params = [gv[1] for gv in grads_and_vars]

            grads = tf.clip_by_global_norm(grads, 5.0)[0]

            # clipped_grads_and_vars = [(
            #    tf.clip_by_norm(grad, 5.0), var) for grad, var in grads_and_vars]
            clipped_grads_and_vars = zip(grads, params)
            self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars,
                global_step=tf.contrib.framework.get_global_step())

            # self.train_op = self.optimizer.minimize(self._loss,
            #    global_step=tf.contrib.framework.get_global_step())
        with tf.name_scope('target_network_update'):
            q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'q_network')]
            q_network_params = sorted(q_network_params, key=lambda v: v.name)

            target_q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'target_q_network')]
            target_q_network_params = sorted(target_q_network_params, key=lambda v: v.name)

            self.target_update_ops = []
            for e1_v, e2_v in zip(q_network_params, target_q_network_params):
                op = e2_v.assign(e1_v)
                self.target_update_ops.append(op)
