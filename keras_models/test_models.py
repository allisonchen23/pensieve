import a3c_keras
import a3c_test
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint

# Taken from sim/agent.py
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NN_MODEL = 'test/models/pretrain_linear_reward.ckpt'

def load_model_to_keras():
    with tf.Session() as sess:

        actor = a3c_keras.ActorNetwork(
            sess=sess,
            state_dim=[S_INFO, S_LEN],
            action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE
        )

        # critic = a3c_keras.CriticNetwork(
        #     sess=sess,
        #     state_dim=[S_INFO, S_LEN],
        #     learning_rate=CRITIC_LR_RATE
        # )

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            reader = tf.train.NewCheckpointReader(nn_model)
            actor.restore_weights(nn_model)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # for key, value in sorted(var_to_shape_map.items()):
            #     if 'RMSProp' not in key and 'critic' not in key:
            #         print("{} {}".format(key, value))
            # print(reader.get_tensor('actor/Conv1D/W'))
            # inspect_checkpoint.print_tensors_in_checkpoint_file(
            #     file_name=nn_model,
            #     tensor_name=None,
            #     all_tensors=True)

            # saver.restore(sess, nn_model)
            print("Model restored.")

def load_model():
    with tf.Session() as sess:

        actor = a3c_test.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c_test.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            # actor.load_weights(NN_MODEL)
            print(actor.get_network_params())
            print("Model restored.")

if __name__=="__main__":

    '''
    Load .ckpt model into Keras
    '''
    load_model_to_keras()
    # load_model()

