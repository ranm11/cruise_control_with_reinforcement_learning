#os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import tensorflow as tf
from ou_Noise_generator import OUActionNoise
import numpy as np

class DDPG_network:
    def __init__(self,num_states,num_actions,lower_bound,upper_bound):
        self.NOF_states = num_states
        self.NOF_actions = num_actions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        self.actor_model =   self.get_actor()
        self.critic_model =  self.get_critic()
        self.target_actor =  self.get_actor()
        self.target_critic = self.get_critic()
        self.target_actor.set_weights( self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_lr = 0.002
        self.actor_lr = 0.001
        self.critic_optimizer = keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(self.actor_lr)
        self.tau = 0.005
        self.gamma = 0.99

    def policy(self, state):
        sampled_actions = keras.ops.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    def update_target(self, target, original, tau):
        target_weights = target.get_weights()
        original_weights = original.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

        target.set_weights(target_weights)
    
    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.NOF_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = keras.Model(inputs, outputs)
        return model
    
    def get_critic(self):
    # State as input
        state_input = layers.Input(shape=(self.NOF_states,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.NOF_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = keras.Model([state_input, action_input], outputs)

        return model

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update( self, state_batch, action_batch,  reward_batch,  next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
