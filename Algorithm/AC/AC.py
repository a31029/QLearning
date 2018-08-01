import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import collections

import time

import random

import gym
import pickle

from pathlib import Path

from FlappyBird.Config import Parameters
from FlappyBird.Util import Index


class AC:
    def __init__(self, scale=1,load = True,state_dim = 3,do_train = False):
        self.index = Index(scale, Parameters.state_x_min, Parameters.state_x_max,Parameters.state_y_min, Parameters.state_y_max)
        self.state_dim = state_dim

        self.gamma = 0.9
        self.load = load
        self.prev = None

        self.do_train = do_train

        self.get_session()
        self.build_net()

    def build_net(self):

        tf.reset_default_graph()

        self.learning_rate_critic = 1e-3
        self.learning_rate_actor = 5e-4

        self.s_eval = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.s_target = tf.placeholder(tf.float32, shape=[None, self.state_dim])

        self.a = tf.placeholder(tf.int32, shape=[None,1])
        self.r = tf.placeholder(tf.float32, shape=[None,1])
        self.td_error = tf.placeholder(tf.float32, shape=[None,1])

        with tf.variable_scope('actor'):
            self.l1_actor = tf.layers.dense(self.s_eval,32,activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            self.l2_actor = tf.layers.dense(self.l1_actor,32,activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            self.output_actor = tf.layers.dense(self.l2_actor,2,activation=tf.nn.softmax,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            

        with tf.variable_scope('critic_eval'):

            self.l1_w_critic_eval = tf.get_variable('l1_w_critic_eval', shape=[self.state_dim, 32])
            self.l1_b_critic_eval = tf.get_variable('l1_b_critic_eval', shape=[32])
            
            self.l2_w_critic_eval = tf.get_variable('l2_w_critic_eval', shape=[32, 16])
            self.l2_b_critic_eval = tf.get_variable('l2_b_critic_eval', shape=[16])
            
            self.l3_w_critic_eval = tf.get_variable('l3_w_critic_eval', shape=[16, 1])
            self.l3_b_critic_eval = tf.get_variable('l3_b_critic_eval', shape=[1,])
            
            self.l1_critic_eval = tf.nn.leaky_relu(tf.matmul(self.s_eval, self.l1_w_critic_eval) + self.l1_b_critic_eval)
            self.l2_critic_eval = tf.nn.leaky_relu(tf.matmul(self.l1_critic_eval, self.l2_w_critic_eval) + self.l2_b_critic_eval)
            self.output_critic_eval = tf.matmul(self.l2_critic_eval, self.l3_w_critic_eval) + self.l3_b_critic_eval


        with tf.variable_scope('critic_target'):

            self.l1_w_critic_target = tf.get_variable('l1_w_critic_target', shape=[self.state_dim, 32])
            self.l1_b_critic_target = tf.get_variable('l1_b_critic_target', shape=[32])
 
            self.l2_w_critic_target = tf.get_variable('l2_w_critic_target', shape=[32, 16])
            self.l2_b_critic_target = tf.get_variable('l2_b_critic_target', shape=[16])
            
            self.l3_w_critic_target = tf.get_variable('l3_w_critic_target', shape=[16, 1])
            self.l3_b_critic_target = tf.get_variable('l3_b_critic_target', shape=[1])

            self.l1_critic_target = tf.nn.leaky_relu(tf.matmul(self.s_target, self.l1_w_critic_target) + self.l1_b_critic_target)
            self.l2_critic_target = tf.nn.leaky_relu(tf.matmul(self.l1_critic_target, self.l2_w_critic_target) + self.l2_b_critic_target)
            self.output_critic_target = tf.matmul(self.l2_critic_target, self.l3_w_critic_target) + self.l3_b_critic_target

        with tf.variable_scope('loss'):

            self.critic_loss = tf.reduce_mean(tf.square(self.output_critic_eval - (self.gamma * self.output_critic_target + self.r)))
            self.log_prob = tf.reduce_sum(tf.multiply(tf.log(self.output_actor),tf.one_hot(tf.reshape(self.a,shape=[-1]),2)),axis = 1)
            self.actor_loss = tf.reduce_mean(tf.multiply(self.log_prob, tf.reshape(self.td_error,shape=[-1])))

        with tf.variable_scope('train'):
            self.op_critic = tf.train.AdamOptimizer(self.learning_rate_critic)
            self.train_critic = self.op_critic.minimize(self.critic_loss)
            self.op_actor = tf.train.AdamOptimizer(self.learning_rate_actor)
            self.train_actor = self.op_actor.minimize(self.actor_loss)

        self.get_session()

        self.saver = tf.train.Saver()

        if len([str(x) for x in Path('Algorithm/AC/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.load:
            self.saver.restore(self.sess, "Algorithm/AC/Net/model.ckpt")
            self.memory = collections.deque(self.loadFile('Algorithm/AC/Memory/memory'), maxlen=5000)
            print('--------------load parameters------------------')
        else:
            print('--------------init---------------------------')
            self.sess.run(tf.global_variables_initializer())
            self.memory = collections.deque([], maxlen=5000)

        self.mem_batch = []

        self.update_params()

        self.train_counter = 0

            
    def get_session(self):
        self.sess = tf.Session()

    def update_params(self):
        self.sess.run(tf.assign(self.l1_w_critic_target, self.l1_w_critic_eval))
        self.sess.run(tf.assign(self.l1_b_critic_target, self.l1_b_critic_eval))
        self.sess.run(tf.assign(self.l2_w_critic_target, self.l2_w_critic_eval))
        self.sess.run(tf.assign(self.l2_b_critic_target, self.l2_b_critic_eval))
        self.sess.run(tf.assign(self.l3_w_critic_target, self.l3_w_critic_eval))
        self.sess.run(tf.assign(self.l3_b_critic_target, self.l3_b_critic_eval))


    def get_action(self, s):
        if self.train_counter < 100:
            return np.random.choice([0,0,0,0,0,0,0,1])
        else:
            output_actor = self.sess.run(self.output_actor, feed_dict={self.s_eval: s})
            return np.random.choice([0,1],p=output_actor[0])

    def ctrain(self):
        for kk in range(150):
            length = min(len(self.memory), 24)
            batch = np.array(self.memory)[np.random.choice(len(self.memory), length)]

            s = batch[:,:self.state_dim]
            a = batch[:, self.state_dim:self.state_dim+1]
            r = batch[:, self.state_dim+1:self.state_dim+2]
            s_ = batch[:, self.state_dim+2:]

            _,c_loss = self.sess.run([self.train_critic,self.critic_loss], feed_dict={self.s_eval: s, self.s_target: s_, self.r: r})

        print('c_loss is ',c_loss)
        
    def atrain(self,data):
        
        s = data[:,:self.state_dim]
        a = data[:, self.state_dim:self.state_dim+1]
        s_ = data[:, self.state_dim+2:]

        old = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s})
        new = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s_})
        v_error = (old - new)
        _,a_loss,log_prob = self.sess.run([self.train_actor,self.actor_loss,self.log_prob], feed_dict={self.a:a,self.s_eval:s,self.td_error: v_error})
        
        # print(v_error[0:30])
        # print(a[0:30])
        # print(log_prob[0:30])
        print('actor_loss is : ',a_loss)


    def run(self, now, dead, action):

        state = now

        action = self.get_action(np.array(state)[np.newaxis,:])

        if dead >= 0 :
            reward = 1
        else:
            reward = -10
        
        if self.prev is None:
            self.prev = np.hstack((state,action))
        else:
            self.prev = np.hstack((self.prev,reward,state))
            self.mem_batch.append(self.prev)
            self.prev = np.hstack((state,action))

        if dead < 0:
            self.train_counter += 1
            tmp = np.array(self.mem_batch)
            tmp[:,self.state_dim+1] = self._discount_and_norm_rewards(tmp[:,self.state_dim+1])
            self.prev = None
            self.memory.extend(tmp.tolist())
            # self.memory.extend(self.mem_batch)
            
            if self.train_counter > 100 and self.do_train:
                self.atrain(tmp)
            
            self.mem_batch = []

            if self.train_counter % 20 == 0 and len(self.memory) > 1000 and self.do_train:
                self.ctrain()

            if self.train_counter % 100 == 0 and self.do_train:
                self.saveNet()
                self.saveFile('Algorithm/AC/Memory/memory', self.memory)
                self.update_params()
                print('update parameters\n')
        return action


    def saveNet(self):
        self.saver.save(self.sess, "Algorithm/AC/Net/model.ckpt")

    def saveFile(self, fileName, obj):
            path = Path(fileName)
            with path.open('wb') as f:
                pickle.dump(obj, f)

    def loadFile(self, fileName):
        path = Path(fileName)
        if path.exists():
            with path.open('rb') as f:
                obj = pickle.load(f)
                return obj
        else:
            return {}



    def _discount_and_norm_rewards(self,v):
        discounted_ep_rs = np.zeros_like(v).astype(float)
        running_add = 0
        for t in reversed(range(0, discounted_ep_rs.shape[0])):
            running_add = running_add * self.gamma + v[t]
            discounted_ep_rs[t] = running_add
        # # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # # print(discounted_ep_rs)
        return discounted_ep_rs