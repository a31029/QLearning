import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
import time
import random
import pickle
from pathlib import Path

from PIL import Image



class AC:
    def __init__(self,l1 = 1e-4,l2=1e-3,state_dim = 4,do_load = False,do_save = False,do_train = False):

        self.learning_rate_critic = l1
        self.learning_rate_actor = l2

        self.state_dim = state_dim
        self.do_load = do_load
        self.do_save = do_save
        self.do_train = do_train

        self.gamma_value = 0.9

        self.train_counter = 1

        self._get_session()
        self._build_net()
        self._build_memory()

        self.max_value = 0


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
            self.l1_actor = tf.layers.dense(self.s_eval,32,activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer(0,0.1))
            self.l2_actor = tf.layers.dense(self.l1_actor,16,activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer(0,0.1))
            self.output_actor = tf.layers.dense(self.l2_actor,2,activation=tf.nn.softmax,kernel_initializer=tf.random_normal_initializer(0,0.1))
            
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

            self.critic_loss = tf.reduce_mean(tf.square(self.output_critic_eval - (self.gamma_value * self.output_critic_target * tf.cast(tf.greater(self.r , 0),tf.float32) + self.r)))
            self.log_prob = tf.log(self.output_actor)
            self.actor_loss = (tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.a,2) * self.log_prob ,axis = 1) * self.td_error * -1))


        with tf.variable_scope('train'):
            self.op_critic = tf.train.AdamOptimizer(self.learning_rate_critic)
            self.train_critic = self.op_critic.minimize(self.critic_loss)
            self.op_actor = tf.train.AdamOptimizer(self.learning_rate_actor)
            self.train_actor = self.op_actor.minimize(self.actor_loss)


        self.saver = tf.train.Saver()

        if len([str(x) for x in Path('Algorithm/AC/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load:
            self.saver.restore(self.sess, "Algorithm/AC/Net/model.ckpt")
            # print('load net parameters!!!')
        else:
            # print('init net parameters!!!')
            self.sess.run(tf.global_variables_initializer())
        self._update()

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []

        self.memory = collections.deque(self.loadFile('Algorithm/AC/Memory/memory'), maxlen=2000)


    def _update(self):
        self.sess.run(tf.assign(self.l1_w_critic_target, self.l1_w_critic_eval))
        self.sess.run(tf.assign(self.l1_b_critic_target, self.l1_b_critic_eval))
        self.sess.run(tf.assign(self.l2_w_critic_target, self.l2_w_critic_eval))
        self.sess.run(tf.assign(self.l2_b_critic_target, self.l2_b_critic_eval))
        self.sess.run(tf.assign(self.l3_w_critic_target, self.l3_w_critic_eval))
        self.sess.run(tf.assign(self.l3_b_critic_target, self.l3_b_critic_eval))
        # print('update target net parameters')

    def _get_action(self, s):
        # if self.train_counter < 400:
        #     return np.random.choice([0,0,0,0,0,0,0,0,1])
        #     # return 0
        # else:
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
        self.memory.extend(self.memory_batch)
        self.memory_batch = []
        self.prev = None

    def run(self, now, dead, action):

        state = np.array([now[0] - now[2] ,now[1] - now[3] ,now[3], now[4]])
        action = self._get_action(state)
        if dead >= 0 :
            reward = 1
        else:
            reward = dead
        self._remember_step(state,action,reward)

        if dead > self.max_value:
            self.max_value = dead

        if dead < 0:
            self._remember_batch()

            if len(self.memory) > 300:
                if self.train_counter % 10  == 0 or self.train_counter <= 20:
                    self.ctrain()
                if self.train_counter > 20:
                    self.atrain()
                if self.train_counter % 20 == 0 :
                    self._update()
                    self.saveFile('Algorithm/AC/Memory/memory', self.memory)
                    self.saveNet()
                self.train_counter += 1
                if self.train_counter % 200 == 0:
                    print('小鸟训练了%s次，最大的数字为%s：'%(self.train_counter,self.max_value))
        return action


    def atrain(self):
        if self.do_train:
            data = self.tmp
            s = data[:,:self.state_dim]
            a = data[:, self.state_dim]
            r = data[:,self.state_dim+1:self.state_dim+2]
            s_ = data[:, self.state_dim+2:]

            old = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s})
            new = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s_})
            v_error = new  + r - old
            _,a_loss,op = self.sess.run([self.train_actor,self.actor_loss,self.output_actor], feed_dict={self.a:a,self.s_eval:s,self.td_error: v_error})
            
            # print(v_error[[-3,-2,-1]])
            # print(data[[-3,-2,-1]])
            print('a_loss: %.2f'%(a_loss))
    
    def ctrain(self):
        if self.do_train:
            for i in range(120):
                length = min(len(self.memory), 32)
                data = np.array(self.memory)[np.random.choice(len(self.memory), length,replace=False)]

                s = data[:,:self.state_dim]
                r = data[:, self.state_dim+1:self.state_dim+2]
                s_ = data[:, self.state_dim+2:]

                _,c_loss = self.sess.run([self.train_critic,self.critic_loss], feed_dict={self.s_eval: s, self.s_target: s_, self.r: r})
            print('c_loss:%.2f'%(c_loss))
            self.plot()
        

    def saveNet(self):
        if self.do_save:
            self.saver.save(self.sess, "Algorithm/AC/Net/model.ckpt")
            # print('net has been saved successfully')

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



    # def _discount_and_norm_rewards(self,v):
    #     discounted_ep_rs = np.zeros_like(v).astype(float)
    #     running_add = 0
    #     for t in reversed(range(0, discounted_ep_rs.shape[0])):
    #         running_add = running_add * self.gamma_value + v[t]
    #         discounted_ep_rs[t] = running_add
    #     # normalize episode rewards
    #     # discounted_ep_rs -= np.mean(discounted_ep_rs)
    #     # discounted_ep_rs /= np.std(discounted_ep_rs)
    #     # print(discounted_ep_rs)
    #     return discounted_ep_rs

    def plot(self):

        xx,yy = np.meshgrid(np.arange(600),np.arange(280))
        x = xx.flatten() - 228
        y = yy.flatten() - 380
        yd = np.ones_like(x) * 360
        sp = np.ones_like(y) * 4.5
        arr = np.vstack((x,y,yd,sp)).transpose(1,0)
        re = self.sess.run(self.output_actor,feed_dict={self.s_eval:arr})
        img = ((re[:,0]-re[:,1])>0).astype(np.int).reshape(280,600).transpose(1,0)
        plt.imsave('FlappyBird/graph/z.png',img,cmap=plt.cm.gray)