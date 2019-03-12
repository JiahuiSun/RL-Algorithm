# coding: utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random


class DeepQLearning:
    def __init__(  # 缺乏对神经网络的定义，比如hiddenlayer大小，可以用**kwargs
        self,
        action_dim,                 # action number
        state_dim,                  # state number
        gamma=0.9,                  # discounted rate
        epsilon=0.5,                # explore rate
        hidden_nodes_size=30,       # hidden layer nodes number
        replay_buffer_size=10000,   # max buffer size
        replace_target_freq=5,      # fix Q-target update steps
        learning_rate=0.01,         # learning rate of DNN
        batch_size=32,              # batch size
        epsilon_decrement=None,     # explore decrease
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.hidden_nodes_size = hidden_nodes_size
        self.replay_buffer = deque()
        self.replay_buffer_size = replay_buffer_size
        self.replace_target_freq = replace_target_freq
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.epsilon_decrement = epsilon_decrement

        self.learn_step_counter = 0
        
        self._build_Q_network()


    def _build_Q_network(self):
        state_inputs = keras.layers.Input(shape=(self.state_dim,))
        # action_inputs = keras.layers.Input(shape=(self.action_dim,)) # one-hot
        hid_1 = keras.layers.Dense(self.hidden_nodes_size, activation='relu')(state_inputs)
        self.q_sa = keras.layers.Dense(self.action_dim)(hid_1)
        # self.q_eval = keras.layers.dot([self.q_sa, action_inputs], axes=1)
        self.eval_model = keras.models.Model(inputs=state_inputs, outputs=self.q_sa)
        self.eval_model.compile(loss='mean_squared_error', 
                                optimizer=tf.keras.optimizers.RMSprop(self.learning_rate),
                                metrics=['mean_squared_error'])

        self.targ_model = self.eval_model
  
    
    # 训练网络
    def train_Q_network(self):
        # 如果到达步数就更新参数
        if self.learn_step_counter % self.replace_target_freq == 0:
            self.eval_model.save('eval_model.h5')
            self.targ_model = keras.models.load_model('eval_model.h5')

        # sample (s, a, r, s')
        try:
            batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch = [data[0] for data in batch]
            action_batch = [data[1] for data in batch]
            reward_batch = [data[2] for data in batch]
            next_state_batch = [data[3] for data in batch]
            done = [data[4] for data in batch]
        except:
            print('no enough buffer history')
            return 
        
        # set y labels
        q_batch = self.targ_model.predict(next_state_batch, batch_size=self.batch_size)
        q_sa_batch = np.dot(q_batch, action_batch) 
        q_target_batch = []
        for i in range(self.batch_size):
            q_target_batch.append(reward_batch + self.gamma * (1 - done) * max(q_batch[i]))
        self.history = self.eval_model.fit([state_batch, action_batch], q_target_batch, epochs=100, batch_size=self.batch_size)
        self.learn_step_counter += 1


    def save_transtion(self, state, action, reward, next_state, done):
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1
        self.replay_buffer.append((state, action_one_hot, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.popleft()

    
    def choose_action(self, observation):
        # make it into batch form
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action = random.randint(0, self.action_dim-1)
        else:
            # forward feed the observation and get q value for every actions
            
            actions_value = self.eval_model.predict(observation)
            action = np.argmax(actions_value)

        if self.epsilon_decrement is not None:
            self.epsilon -= self.epsilon_decrement
        
        return action

