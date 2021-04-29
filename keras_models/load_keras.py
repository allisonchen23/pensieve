import a3c_keras
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint

# Taken from sim/agent.py
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# NN_MODEL = 'test/models/pretrain_linear_reward.ckpt'
NN_MODEL = 'sim/results/pretrain_linear_reward.ckpt'

def load_model_to_keras(ckpt_path=NN_MODEL,
                        h5_save_path=None,
                        csv_save_dir=None,
                        state_info=S_INFO,
                        state_len=S_LEN,
                        actor_dim=A_DIM,
                        actor_lr=ACTOR_LR_RATE,
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
    Create actor model in keras, load in a ckpt checkpoint, and save as h5
        Arg(s):
            ckpt_path : str
                path to ckpt file
            h5_save_path : None or str
                path to save h5 model in. If None, stores in same path as ckpt_path, but with .h5 extension
            csv_save_dir : None or str
                if None, do not save weights as CSV. Otherwise save weights as CSV files in this directory
            state_info : int
                number of variables in state
            state_len : int
                length of state
            actor_dim : int
                number of dimensions in actor
            actor_lr : float
                learning rate for actor
            ckpt_layer_names : list[str]
                list of names of ckpt layers (not including '/W' or '/b' for weights and biases)
            keras_layer_names : list[str]
                list of names of corresponding h5 layers
        Returns:
            actor model in Keras
    '''
    with tf.Session() as sess:

        actor = a3c_keras.ActorNetwork(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if ckpt_path is not None:  # nn_model is the path to file
            actor.load_ckpt_store_h5(
                ckpt_path=ckpt_path,
                save_path=h5_save_path,
                csv_save_dir=csv_save_dir,
                ckpt_layer_names=ckpt_layer_names,
                keras_layer_names=keras_layer_names
                )
            print("Model restored.")

        # Return saved model
        return actor.model

if __name__=="__main__":

    '''
    Load .ckpt model into Keras
    '''
    load_model_to_keras(
        csv_save_dir="keras_models/pensieve_csv"
    )

