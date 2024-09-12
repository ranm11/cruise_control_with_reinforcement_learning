import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import  gym
import numpy as np
import matplotlib.pyplot as plt
from lane_gym import CarLaneTrackingEnv
from replay_buffer import Buffer
from DDPG_network import DDPG_network
enable_lane_curvature = True
speed_update_enable = False
env = CarLaneTrackingEnv(enable_lane_curvature)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

#hyperParameter 
# batchSize in replay_buffer.batch_size = 64
total_episodes = 900
# Discount factor for future rewards

# Used to update target networks


buffer = Buffer(num_states,num_actions,50000, 64)
ddpg = DDPG_network(num_states,num_actions,lower_bound,upper_bound)

ep_reward_list = []
avg_reward_list = []
for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    i=0
    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = ddpg.policy(tf_prev_state)
        state, reward, done, _ = env.step(action)
        env.render()
        if(speed_update_enable):
            current_speed = state[2]*env.speedNormalizeFactor
            updated_speed = (current_speed + 0.1) if current_speed < 27 else 28
            env.SpeedUpdate(updated_speed)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward
        #get mini batch K 
        state_batch, action_batch, reward_batch, next_state_batch  = buffer.get_mini_batch()
        ddpg.update(state_batch, action_batch, reward_batch, next_state_batch )
        # learn network
        ddpg.update_target(ddpg.target_actor, ddpg.actor_model, ddpg.tau)
        ddpg.update_target(ddpg.target_critic, ddpg.critic_model, ddpg.tau)
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)