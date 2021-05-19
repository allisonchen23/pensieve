'''
Keras implementation of pensieve/sim/a3c.py
'''
import os
import numpy as np
import tensorflow as tf
import tflearn
import tensorflow.contrib.keras as keras
from keras.layers import Input, Dense, concatenate
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Model

GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 4


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out, self.model = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            # Generate inputs
            inputs_0 = Input(shape=(1,))
            inputs_1 = Input(shape=(1,))
            inputs_2 = Input(shape=(1, self.s_dim[1]))
            inputs_3 = Input(shape=(1, self.s_dim[1]))
            inputs_4 = Input(shape=(1, A_DIM))
            inputs_5 = Input(shape=(1,))

            inputs_list = [inputs_0, inputs_1, inputs_2, inputs_3, inputs_4, inputs_5]

            # Create layers
            split_0 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(inputs_0)
            split_1 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(inputs_1)
            split_2 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_2)
            split_3 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_3)
            split_4 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_4)
            split_5 = Dense(128,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_5)

            # Flatten the convolutional layers
            split_2_flat = Flatten()(split_2)
            split_3_flat = Flatten()(split_3)
            split_4_flat = Flatten()(split_4)

            # Merge all layers to be "parallel" inputs
            merge_net = concatenate([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=-1)
            # merge_net = concatenate([split_0, split_1, split_2_flat, split_3_flat, split_5], axis=0)

            # Add a fully connected layer
            dense_net_0 = Dense(
                128,
                activation='relu',
                kernel_initializer='truncated_normal')(merge_net)

            # Create outputs
            out = Dense(
                self.a_dim,
                activation='softmax',
                kernel_initializer='truncated_normal')(dense_net_0)

            model = Model(inputs=inputs_list, outputs=out)

            return inputs_list, out, model

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def load_ckpt_store_h5(self,
                        ckpt_path,
                        save_path=None,
                        csv_save_dir=None,
                        ckpt_layer_names=[
                            'actor/FullyConnected',
                            'actor/FullyConnected_1',
                            'actor/Conv1D',
                            'actor/Conv1D_1',
                            'actor/Conv1D_2',
                            'actor/FullyConnected_2',
                            'actor/FullyConnected_3',
                            'actor/FullyConnected_4'
                            ],
                        keras_layer_names = [
                            'dense_1',
                            'dense_2',
                            'conv1d_1',
                            'conv1d_2',
                            'conv1d_3',
                            'dense_3',
                            'dense_4',
                            'dense_5'
                            ]):
        '''
        Restore weights of Keras model from CKPT file and save as h5 file
        Arg(s):
            ckpt_path : str
                path to ckpt file
            save_path : None or str
                path to save h5 model in. If None, stores in same path as ckpt_path, but with .h5 extension
            csv_save_dir : None or str
                if None, do not save weights as CSV. Otherwise save weights as CSV files in this directory
            ckpt_layer_names : list[str]
                list of names of ckpt layers (not including '/W' or '/b' for weights and biases)
            keras_layer_names : list[str]
                list of names of corresponding h5 layers
        Returns:
            None
        '''

        # Checks for paths
        if save_path is None:
            save_path = ckpt_path.replace('.ckpt', '.h5')

        if csv_save_dir is not None:
            # Clear directory if it exists
            if os.path.isdir(csv_save_dir):
                os.system("rm -rf {}".format(csv_save_dir))
            os.makedirs(csv_save_dir)

        assert len(ckpt_layer_names) == len(keras_layer_names)

        # Obtain reader to understand ckpt checkpoints
        reader = tf.train.NewCheckpointReader(ckpt_path)

        for layer_idx, (ckpt_layer_name, keras_layer_name) in enumerate(zip(ckpt_layer_names, keras_layer_names)):
            # Obtain old weights (to check shape later)
            old_weights = self.model.get_layer(keras_layer_name).get_weights()

            # Extract weight values from ckpt layers and store into keras layers
            weights = reader.get_tensor(ckpt_layer_name + '/W')

            # Assume that one dimension is a 1 and can be squeezed to match
            if old_weights[0].shape != weights.shape:
                weights = np.squeeze(weights)
                assert old_weights[0].shape == weights.shape

            biases = reader.get_tensor(ckpt_layer_name + '/b')


            self.model.get_layer(keras_layer_name).set_weights([weights, biases])

            new_weights = self.model.get_layer(keras_layer_name).get_weights()

            # Sanity check
            for old_weight, new_weight in zip(old_weights, new_weights):
                assert not (old_weight == new_weight).all()

            # Save to CSV, if desired
            if csv_save_dir is not None:
                '''
                Save weights
                '''

                # Check if 3d tensor
                if 2 < len(weights.shape):
                    # iterate through each kernel
                    for kernel_idx in range(weights.shape[0]):
                        kernel_weights = weights[kernel_idx]
                        np.savetxt(
                            os.path.join(csv_save_dir, "weights_layer{}_kernel{}.csv".format(layer_idx, kernel_idx)),
                            kernel_weights,
                            delimiter=",")
                else:
                    np.savetxt(
                        os.path.join(csv_save_dir, "weights_layer{}.csv".format(layer_idx)),
                        weights,
                        delimiter=",")

                '''
                Save biases
                '''
                np.savetxt(
                    os.path.join(csv_save_dir, "bias_layer{}.csv".format(layer_idx)),
                        biases,
                        delimiter=",")


        print("Saving model to {}".format(save_path))
        self.model.save(save_path)

    def save_actor_end(self,
                       save_h5_path=None,
                       save_csv_dir=None):
        '''
        Create and return ActorNetworkEnd (only the last 2 layers) to feed into COMET
        Arg(s):
            save_h5_path : str or None
                save .h5 model to path if specified
            save_csv_dir : str or None
                save weights as CSV files in this directory if specified
        Returns:
            ActorNetworkEnd object
        '''
        self.model.summary()
        actor_end = ActorNetworkEnd(self.sess, self.s_dim, self.a_dim, self.lr_rate)
        actor_end.model.summary()
        actor_layer_names = ['dense_4', 'dense_5']
        actor_end_layer_names = ['dense_1', 'dense_2']
        if save_csv_dir is not None:
            if os.path.isdir(save_csv_dir):
                os.system("rm -rf {}".format(save_csv_dir))
            os.makedirs(save_csv_dir)
        if save_h5_path is not None:
            h5_dir = os.path.dirname(save_h5_path)
            if not os.path.isdir(h5_dir):
                os.makedirs(h5_dir)

        for idx, (actor_layer_name, actor_end_layer_name) in enumerate(zip(actor_layer_names, actor_end_layer_names)):
            weights = self.model.get_layer(actor_layer_name).get_weights()
            print(actor_end_layer_name)
            actor_end.model.get_layer(actor_end_layer_name).set_weights(weights)

            if save_csv_dir is not None:
                np.savetxt(
                    os.path.join(save_csv_dir, "weights_layer{}.csv".format(layer_idx)),
                    weights,
                    delimiter=",")

        if save_h5_path is not None:
            actor_end.model.save(save_h5_path)

        return actor_end

    def load_weights(self, h5_path):
        '''
        Load weights into actor model
        Arg(s):
            h5_path : str
                path to .h5 model weights
        '''
        self.model = keras.models.load_model(h5_path)


class ActorNetworkEnd(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out, self.model = self.create_actor_network_end()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network_end(self):
        with tf.variable_scope('actor'):
            # Generate inputs

            inputs = Input(shape=(768,))

            inputs_list = [inputs]
            # Create layers
            # Add a fully connected layer
            dense_net_0 = Dense(
                128,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs)

            # Create outputs
            out = Dense(
                self.a_dim,
                activation='softmax',
                kernel_initializer='truncated_normal')(dense_net_0)

            model = Model(inputs=inputs_list, outputs=out)

            return inputs_list, out, model

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def get_summary(self):
        self.model.summary()
        print("Layers: {}".format([layer.name for layer in self.model.layers]))

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            # Generate inputs
            inputs = Input(shape=(self.s_dim[0], self.s_dim[1]))
            inputs_0 = inputs[:, 0:1, -1]
            inputs_1 = inputs[:, 1:2, -1]
            inputs_2 = inputs[:, 2:3, :]
            inputs_3 = inputs[:, 3:4, :]
            inputs_4 = inputs[:, 4:5, :A_DIM]
            inputs_5 = inputs[:, 4:5, -1]

            # Create layers
            split_0 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(inputs_0)
            split_1 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(inputs_1)
            split_2 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_2)
            split_3 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal')(inputs_3)
            split_4 = Conv1D(
                filters=128,
                padding='same',
                kernel_size=4,
                activation='relu',
                kernel_initializer='truncated_normal', use_bias=False)(inputs_4)
            split_5 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(inputs_5)

            # Flatten the convolutional layers
            split_2_flat = Flatten()(split_2)
            split_3_flat = Flatten()(split_3)
            split_4_flat = Flatten()(split_4)

            # Merge all layers to be "parallel" inputs
            merge_net = concatenate([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=0)

            # Add a fully connected layer
            dense_net_0 = Dense(
                128,
                activation='relu',
                kernel_initializer='truncated_normal')(merge_net)

            # Create outputs
            out = Dense(
                1,
                activation='linear',
                kernel_initializer='truncated_normal')(dense_net_0)

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(xrange(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in xrange(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars