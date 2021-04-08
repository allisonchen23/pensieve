import a3c_keras
import tensorflow as tf

# Taken from sim/agent.py
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001

if __name__=="__main__":
    actor = a3c_keras.ActorNetwork(
        sess=tf.Session(),
        state_dim=[S_INFO, S_LEN],
        action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE
    )

    critic = a3c_keras.CriticNetwork(
        sess=tf.Session(),
        state_dim=[S_INFO, S_LEN],
        learning_rate=ACTOR_LR_RATE
    )