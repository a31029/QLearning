import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import collections
import time
import random
import pickle
from pathlib import Path



class AC:
    def __init__(self,state_dim = 4,do_load = False,do_save = False,do_train = False):

        self.state_dim = state_dim
        self.do_load = do_load
        self.do_save = do_save
        self.do_train = do_train

        self.gamma_value = 0.9
        self.gamma_reward = 0

        self.train_counter = 1

        self._get_session()
        self._build_net()
        self._build_memory()


    def _get_session(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

    def _build_net(self):

        self.s_eval = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.s_target = tf.placeholder(tf.float32, shape=[None, self.state_dim])

        self.a = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None,1])
        self.td_error = tf.placeholder(tf.float32, shape=[None,1])

        with tf.variable_scope('actor'):
            self.l1_actor = tf.layers.dense(self.s_eval,32,activation=tf.nn.tanh)
            self.l2_actor = tf.layers.dense(self.l1_actor,32,activation=tf.nn.tanh)
            self.output_actor = tf.layers.dense(self.l2_actor,2,activation=tf.nn.softmax)
            
        with tf.variable_scope('critic_eval'):

            self.l1_w_critic_eval = tf.get_variable('l1_w_critic_eval', shape=[self.state_dim, 32])
            self.l1_b_critic_eval = tf.get_variable('l1_b_critic_eval', shape=[32])
            
            self.l2_w_critic_eval = tf.get_variable('l2_w_critic_eval', shape=[32, 16])
            self.l2_b_critic_eval = tf.get_variable('l2_b_critic_eval', shape=[16])
            
            self.l3_w_critic_eval = tf.get_variable('l3_w_critic_eval', shape=[16, 1])
            self.l3_b_critic_eval = tf.get_variable('l3_b_critic_eval', shape=[1])
            
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

            self.critic_loss = tf.reduce_sum(tf.square(self.output_critic_eval - (self.gamma_value * self.output_critic_target * tf.cast(tf.greater(self.r , 0),tf.float32) + self.r)))
            self.log_prob = tf.log(self.output_actor)
            self.actor_loss = tf.reduce_sum(tf.reduce_sum(tf.one_hot(self.a,2) * self.log_prob ,axis = 1) * self.td_error)

        self.learning_rate_critic = 1e-3
        self.learning_rate_actor = 5e-5

        with tf.variable_scope('train'):
            self.op_critic = tf.train.AdamOptimizer(self.learning_rate_critic)
            self.train_critic = self.op_critic.minimize(self.critic_loss)
            self.op_actor = tf.train.AdamOptimizer(self.learning_rate_actor)
            self.train_actor = self.op_actor.minimize(self.actor_loss)


        self.saver = tf.train.Saver()

        if len([str(x) for x in Path('Algorithm/AC/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load:
            self.saver.restore(self.sess, "Algorithm/AC/Net/model.ckpt")
            print('load net parameters!!!')
        else:
            print('init net parameters!!!')
            self.sess.run(tf.global_variables_initializer())
        self._update()

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []

        self.memory = collections.deque(self.loadFile('Algorithm/AC/Memory/memory'), maxlen=5000)


    def _update(self):
        self.sess.run(tf.assign(self.l1_w_critic_target, self.l1_w_critic_eval))
        self.sess.run(tf.assign(self.l1_b_critic_target, self.l1_b_critic_eval))
        self.sess.run(tf.assign(self.l2_w_critic_target, self.l2_w_critic_eval))
        self.sess.run(tf.assign(self.l2_b_critic_target, self.l2_b_critic_eval))
        self.sess.run(tf.assign(self.l3_w_critic_target, self.l3_w_critic_eval))
        self.sess.run(tf.assign(self.l3_b_critic_target, self.l3_b_critic_eval))
        print('update target net parameters')

    def _get_action(self, s):
        if self.train_counter < 3:
            return np.random.choice([0,0,0,0,0,0,0,1])
        else:
            output_actor = self.sess.run(self.output_actor, feed_dict={self.s_eval: np.array([s])})
            return np.random.choice([0,1],p=output_actor[0])

    def _remember_step(self,state,action,reward):
        if self.prev  is None:
            self.prev = np.hstack((state,action))
        else:
            self.memory_batch.append(np.hstack((self.prev, reward, state)))
            self.prev = np.hstack((state,action))

    def _remember_batch(self):
        self.tmp = np.array(self.memory_batch)
        # self.tmp[:,self.state_dim+1] = self._discount_and_norm_rewards(self.tmp[:,self.state_dim+1])
        self.memory.extend(self.tmp.tolist())
        self.memory_batch = []
        self.prev = None

    def run(self, now, dead, action):

        state = np.array(now)[1:]
        # state = now
        action = self._get_action(state)
        if dead >= 0 :
            reward = 1
        else:
            reward = dead
        
        self._remember_step(state,action,reward)

        if dead < 0:
            self._remember_batch()

            if len(self.memory) > 2000:
                self.train_counter += 1
                if self.train_counter % 3 == 0 or (self.train_counter <= 3):
                    self.ctrain()
                    self.saveFile('Algorithm/AC/Memory/memory', self.memory)
                    self.saveNet()
                    
                if self.train_counter > 3:
                    self.atrain()
                if (self.train_counter % 6 == 0 and self.train_counter > 3):
                    self._update()
        return action


    def atrain(self):
        if self.do_train:
            data = self.tmp
            s = data[:,:self.state_dim]
            a = data[:, self.state_dim]
            s_ = data[:, self.state_dim+2:]

            old = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s})
            new = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s_})
            v_error = (new - old)
            _,a_loss,op = self.sess.run([self.train_actor,self.actor_loss,self.output_actor], feed_dict={self.a:a,self.s_eval:s,self.td_error: v_error})
            
            new_tar = self.sess.run(self.output_critic_target,feed_dict={self.s_target:s_})

            # print(np.hstack((op,v_error,a,old,new,new_tar))[[[-5,-4,-3,-2,-1]]])
            # print(data[[-5,-4,-3,-2,-1]])
            # print('a_loss: %.2f'%(a_loss))
    
    def ctrain(self):
        if self.do_train:
            for i in range(100):
                length = min(len(self.memory), 32)
                data = np.array(self.memory)[np.random.choice(len(self.memory), length,replace=False)]

                s = data[:,:self.state_dim]
                r = data[:, self.state_dim+1:self.state_dim+2]
                s_ = data[:, self.state_dim+2:]

                _,c_loss = self.sess.run([self.train_critic,self.critic_loss], feed_dict={self.s_eval: s, self.s_target: s_, self.r: r})
            print('c_loss:%.2f'%(c_loss))
        

    def saveNet(self):
        if self.do_save:
            self.saver.save(self.sess, "Algorithm/AC/Net/model.ckpt")
            print('net has been saved successfully')

    def saveFile(self, fileName, obj):
        if self.do_save:
            path = Path(fileName)
            with path.open('wb') as f:
                pickle.dump(obj, f)

    def loadFile(self, fileName):
        path = Path(fileName)
        if path.exists() and self.do_load:
            with path.open('rb') as f:
                obj = pickle.load(f)
                return obj
        else:
            return {}


