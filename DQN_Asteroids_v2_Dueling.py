import numpy as np
import gym
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from rl.callbacks import Callback, FileLogger
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


## Step 1 : Create the environment

env = gym.make("Asteroids-v0").env
env.seed(3)
env.reset()

nb_actions = env.action_space.n

# Next, we build a neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(3, input_dim=1,activation= 'tanh'))
model.add(Dense(nb_actions))
model.add(Activation('sigmoid')

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=-1, value_test=.05,
                              nb_steps=1000000)

memory = SequentialMemory(limit=10000000, window_length=1)
dqn2 = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
              target_model_update=1e-2, policy=policy, enable_double_dqn=True, enable_dueling_network=True)
dqn2.compile(Adam(lr=1e-3), metrics=['mae', 'acc'])

import os.path

file_path = 'Dueling_DQN_Taxi.h5f'
if os.path.exists(file_path):
    dqn2.load_weights(file_path)


class Saver(Callback):
    def on_episode_end(self, episode, logs={}):
        print('episode callback')
        if episode % 1 == 0:
            self.model.save_weights('Dueling_DQN_Taxi.h5f', overwrite=True)


logs = FileLogger('Dueling_DQN_Taxi.csv', interval = 1)
s = Saver()
dqn2.fit(env, nb_steps=2e8,callbacks=[s,logs], visualize=False, verbose=2)
#dqn2.test(env, nb_episodes=10, visualize=True)
