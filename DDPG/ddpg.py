# coding: utf-8
import tensorflow as tf
import numpy as np
import gym
import time, sys


class DDPG(object):
    def __init__(self,
                 s_dim,
                 a_dim,
                 r_dim,
                 gamma=0.001,
                 tau=0.0001,
                 lr_c=0.001,
                 lr_a=0.001,
                 var=0.001,
                 batch_size=32,
                 replay_buffer_size=100000,
                 actor_layers=[100, 100],
                 critic_layers=[100, 100],
                 ):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.r_dim = r_dim
        self.gamma = gamma
        self.tau = tau
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.var = var
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.replay_buffer = np.zeros((replay_buffer_size, s_dim*2+a_dim+r_dim), dtype=np.float32)
        self.buffer_ptr = 0
        self.S = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 'next state')
        self.R = tf.placeholder(tf.float32, [None, r_dim], 'reward')

        with tf.variable_scope('Actor'):
            self.a = self.build_actor(self.S, layers=actor_layers,
                                      scope='eval', trainable=True)
            self.a_ = self.build_actor(self.S_, layers=actor_layers,
                                       scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            self.q = self.build_critic(self.S, self.a, layers=critic_layers,
                                       scope='eval', trainable=True)
            self.q_ = self.build_critic(self.S_, self.a_, layers=critic_layers,
                                        scope='target', trainable=False)

        self.actor_eval_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval'
        )
        self.actor_target_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target'
        )
        self.critic_eval_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval'
        )
        self.critic_target_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target'
        )

        self.soft_replace = [[tf.assign(at, (1-self.tau)*at+self.tau*ae),
                              tf.assign(ct, (1-self.tau)*ct+self.tau*ce)]
                             for at, ae, ct, ce in zip(
                                 self.actor_target_params,
                                 self.actor_eval_params,
                                 self.critic_target_params,
                                 self.critic_eval_params)]
        # critic update
        self.q_target = self.R + self.gamma*self.q_
        self.td_error = tf.losses.mean_squared_error(labels=self.q_target,
                                                     predictions=self.q)
        self.critic_train = tf.train.AdamOptimizer(self.lr_c).minimize(
            self.td_error, var_list=self.critic_eval_params
        )
        # actor update
        self.actor_loss = - tf.reduce_mean(self.q)
        self.actor_train = tf.train.AdamOptimizer(self.lr_a).minimize(
            self.actor_loss, var_list=self.actor_eval_params
        )
        # plot graph
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./logs', self.sess.graph)

    def build_actor(self, x, layers, scope, trainable):
        """
        Args:
            x: 输入
            layers: 网络结构 [in, 100, 20, out(1)]，不包括最后输出层
            scope: 命名空间
            trainable: 是否可训练
        """
        with tf.variable_scope(scope):
            for i, layer in enumerate(layers):
                name = 'layer' + str(i)
                x = tf.layers.dense(x, layer,
                                    activation=tf.nn.relu,
                                    name=name, trainable=trainable)
            output = tf.layers.dense(x, 1, activation=tf.nn.tanh,
                                     name='output', trainable=trainable)
            return output

    def build_critic(self, s, a, layers, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable("w1_s", [s.shape[1], layers[0]],
                                   trainable=trainable)
            w1_a = tf.get_variable("w1_a", [a.shape[0], layers[0]],
                                   trainable=trainable)
            b1 = tf.get_variable("b1", [1, layers[0]], trainable=trainable)
            x = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            for i in range(1, len(layers)):
                name = 'layer' + str(i)
                x = tf.layers.dense(x, layers[i], activation=tf.nn.relu,
                                    name=name, trainable=trainable)
            output = tf.layers.dense(x, 1, activation=tf.nn.sigmoid,
                                     name='output', trainable=trainable)
        return output

    def choose_action(self, transition):
        a = self.sess.run(self.a, feed_dict={self.S: transition[np.newaxis, :]})
        action = np.clip(np.random.normal(a, self.var), -1, 1)
        return action

    def save_transition(self, state, action, reward, next_state, done):
        if self.buffer_ptr > 0 and \
           self.buffer_ptr % self.replay_buffer_size == 0:
            self.buffer_ptr = 0
        transition = np.hstack((state, action, reward, next_state, done))
        self.replay_buffer[self.buffer_ptr, :] = transition
        self.buffer_ptr += 1

    def learn(self):
        self.indices = np.random.choice(self.replay_buffer_size,
                                        size=self.batch_size)
        trans_batch = self.replay_buffer[self.indices, :]
        obs_batch = transitions[:, :self.s_dim]
        action_batch = transitions[:, self.s_dim:self.s_dim+self.a_dim]
        reward_batch = transitions[:, self.s_dim+self.a_dim:self.s_dim+self.a_dim+self.r_dim]
        new_obs_batch = transitions[:, -self.s_dim-1:-1]
        # 更新critic
        self.sess.run(self.critic_train,
            feed_dict={
                self.S = obs_batch,
                self.
                self.R = reward_batch,
                
            }
        )
        pass
