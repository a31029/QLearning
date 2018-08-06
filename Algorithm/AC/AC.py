import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
import time
import random
import pickle
from pathlib import Path


class AC:
    def __init__(self,l1 = 1e-4,l2=1e-3,state_dim = 4,do_load = False,do_save = False,do_train = False):

        self.learning_rate_critic = l1
        self.learning_rate_actor = l2

        self.state_dim = state_dim
        self.do_load = do_load
        self.do_save = do_save
        self.do_train = do_train

        self.gamma_value = 0.9

        self.train_counter = 0

        self._get_session()
        self._build_net()
        self._build_memory()

        self.max_value = 0

        print('start AC algorithm')

    def _get_session(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

    def _build_net(self):

        self.s_eval = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.s_target = tf.placeholder(tf.float32, shape=[None, self.state_dim])

        self.a = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None])
        self.td_error = tf.placeholder(tf.float32, shape=[None])

        with tf.variable_scope('actor'):
            self.l1_w_actor = tf.get_variable('l1_w_actor',shape=[self.state_dim,32])
            self.l1_b_actor = tf.get_variable('l1_b_actor',shape=[32,])
            self.l2_w_actor = tf.get_variable('l2_w_actor',shape=[32,16])
            self.l2_b_actor = tf.get_variable('l2_b_actor',shape=[16,])
            self.l3_w_actor = tf.get_variable('l3_w_actor',shape=[16,2])
            self.l3_b_actor = tf.get_variable('l3_b_actor',shape=[2,])

            self.l1_actor = tf.nn.tanh(tf.matmul(self.s_eval,self.l1_w_actor) + self.l1_b_actor)
            self.l2_actor = tf.nn.tanh(tf.matmul(self.l1_actor,self.l2_w_actor)+self.l2_b_actor)
            self.output_actor = tf.nn.softmax(tf.matmul(self.l2_actor,self.l3_w_actor) + self.l3_b_actor)


        with tf.variable_scope('critic_eval'):

            self.l1_w_critic_eval = tf.get_variable('l1_w_critic_eval', shape=[self.state_dim, 32])
            self.l1_b_critic_eval = tf.get_variable('l1_b_critic_eval', shape=[32])
            self.l2_w_critic_eval = tf.get_variable('l2_w_critic_eval', shape=[32, 16])
            self.l2_b_critic_eval = tf.get_variable('l2_b_critic_eval', shape=[16])
            self.l3_w_critic_eval = tf.get_variable('l3_w_critic_eval', shape=[16, 2])
            self.l3_b_critic_eval = tf.get_variable('l3_b_critic_eval', shape=[2])
            
            self.l1_critic_eval = tf.nn.leaky_relu(tf.matmul(self.s_eval, self.l1_w_critic_eval) + self.l1_b_critic_eval)
            self.l2_critic_eval = tf.nn.leaky_relu(tf.matmul(self.l1_critic_eval, self.l2_w_critic_eval) + self.l2_b_critic_eval)
            self.output_critic_eval = tf.matmul(self.l2_critic_eval, self.l3_w_critic_eval) + self.l3_b_critic_eval


        with tf.variable_scope('critic_target'):

            self.l1_w_critic_target = tf.get_variable('l1_w_critic_target', shape=[self.state_dim, 32])
            self.l1_b_critic_target = tf.get_variable('l1_b_critic_target', shape=[32])
            self.l2_w_critic_target = tf.get_variable('l2_w_critic_target', shape=[32, 16])
            self.l2_b_critic_target = tf.get_variable('l2_b_critic_target', shape=[16])
            self.l3_w_critic_target = tf.get_variable('l3_w_critic_target', shape=[16, 2])
            self.l3_b_critic_target = tf.get_variable('l3_b_critic_target', shape=[2])

            self.l1_critic_target = tf.nn.leaky_relu(tf.matmul(self.s_target, self.l1_w_critic_target) + self.l1_b_critic_target)
            self.l2_critic_target = tf.nn.leaky_relu(tf.matmul(self.l1_critic_target, self.l2_w_critic_target) + self.l2_b_critic_target)
            self.output_critic_target = tf.matmul(self.l2_critic_target, self.l3_w_critic_target) + self.l3_b_critic_target


        with tf.variable_scope('loss'):

            self.critic_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_sum(self.output_critic_eval * tf.one_hot(self.a,2),axis = 1)
                     - ( self.gamma_value * tf.reduce_max(self.output_critic_target,axis = 1) * tf.cast(tf.greater(self.r , 0),tf.float32) + self.r)
                )
            )
            self.log_prob = self.output_actor
            self.actor_loss = (tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.a,2) * self.log_prob ,axis = 1) * self.td_error))


        with tf.variable_scope('train'):
            self.op_critic = tf.train.AdamOptimizer(self.learning_rate_critic)
            self.train_critic = self.op_critic.minimize(self.critic_loss)

            self.op_actor = tf.train.AdamOptimizer(self.learning_rate_actor * -1)
            self.train_actor = self.op_actor.minimize(self.actor_loss)


        self.saver = tf.train.Saver()

        if len([str(x) for x in Path('Algorithm/AC/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load:
            # print('load net parameters!!!')
            self.saver.restore(self.sess, "Algorithm/AC/Net/model.ckpt")
        else:
            # print('init net parameters!!!')
            self.sess.run(tf.global_variables_initializer())
        self._update()

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []
        self.memory = collections.deque(self.loadFile('Algorithm/AC/Memory/memory'), maxlen=2000)
        self.memory_actor = collections.deque([],maxlen=50)


    def _update(self):
        self.sess.run(tf.assign(self.l1_w_critic_target, self.l1_w_critic_eval))
        self.sess.run(tf.assign(self.l1_b_critic_target, self.l1_b_critic_eval))
        self.sess.run(tf.assign(self.l2_w_critic_target, self.l2_w_critic_eval))
        self.sess.run(tf.assign(self.l2_b_critic_target, self.l2_b_critic_eval))
        self.sess.run(tf.assign(self.l3_w_critic_target, self.l3_w_critic_eval))
        self.sess.run(tf.assign(self.l3_b_critic_target, self.l3_b_critic_eval))


    def _get_action(self, s):
        output_actor = self.sess.run(self.output_actor, feed_dict={self.s_eval: np.array([s])})
        if np.random.uniform() <= output_actor[0][0]:
            return 0
        else:
            return 1


    def _remember_step(self,state,action,reward):
        if self.prev  is None:
            self.prev = np.hstack((state,action))
        else:
            self.memory_batch.append(np.hstack((self.prev, reward, state)))
            self.prev = np.hstack((state,action))

    def _remember_batch(self):
        self.memory.extend(self.memory_batch)
        self.memory_actor.append(self.memory_batch)
        self.memory_batch = []
        self.prev = None

    def run(self, now, dead, action):

        state = np.array([now[0] - now[2] ,now[1] - now[3] ,now[3], now[4]])
        action = self._get_action(state)
        if dead > self.max_value:
            self.max_value = dead
        if dead >= 0 :
            reward = 1
        else:
            reward = dead

        if state[-1] < 0 and dead >= 0:
            return 0
        else:
            self._remember_step(state,action,reward)

            if dead < 0:
                self._remember_batch()

                if len(self.memory) > 100:
                    self.train_counter += 1
                    if self.train_counter % 3  == 0:
                        self.ctrain()
                    if self.train_counter > 6:
                        self.atrain()
                    if self.train_counter % 6 == 0:
                        self._update()
                        self.saveFile('Algorithm/AC/Memory/memory', self.memory)
                        self.saveNet()
                    if self.train_counter % 6 == 0:
                        print('小鸟训练了%s次，最大的数字为%s：'%(self.train_counter,self.max_value))
            return action


    def atrain(self):
        if self.do_train:
            for i in range(50):
                data = np.array(self.memory_actor[np.random.choice(len(self.memory_actor))])
                # data = np.array(self.memory_actor[-1])
                s = data[:,:self.state_dim]
                a = data[:, self.state_dim]
                r = data[:,self.state_dim + 1]    
                s_ = data[:, self.state_dim+2:]
                old = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s})
                new = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval: s_})
                a_new = self.sess.run(self.output_actor,feed_dict={self.s_eval:s_})

                v_error = self._discount_and_norm_rewards(
                    self.gamma_value * np.sum(new * a_new,axis = 1) + r - old[np.arange(a.shape[0]),a]
                )
                _,a_loss = self.sess.run([self.train_actor,self.actor_loss], feed_dict={self.a:a,self.s_eval:s,self.td_error: v_error})
            
            print('a_loss: %.2f'%(a_loss))
            # print(v_error[-10:])
        
    def ctrain(self):
        if self.do_train:
            for i in range(120):
                length = min(len(self.memory), 32)
                data = np.array(self.memory)[np.random.choice(len(self.memory), length,replace=False)]

                s = data[:,:self.state_dim]
                a = data[:, self.state_dim]
                r = data[:, self.state_dim+1]
                s_ = data[:, self.state_dim+2:]

                _,c_loss = self.sess.run([self.train_critic,self.critic_loss], feed_dict={self.s_eval: s, self.s_target: s_, self.r: r,self.a:a})
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


    def plot(self):

        xx,yy = np.meshgrid(np.arange(600),np.arange(280))
        x = xx.flatten() - 228
        y = yy.flatten() - 380
        yd = np.ones_like(x) * 360
        sp = np.ones_like(y) * 4.5
        arr = np.vstack((x,y,yd,sp)).transpose(1,0)
        # re = self.sess.run(self.output_actor,feed_dict={self.s_eval:arr})
        re = self.sess.run(self.output_critic_eval,feed_dict={self.s_eval:arr})
        img = ((re[:,0]-re[:,1])>0).astype(np.int).reshape(280,600).transpose(1,0)
        plt.imsave('FlappyBird/graph/z.png',img,cmap=plt.cm.gray)


    def _discount_and_norm_rewards(self,v):
        discounted_ep_rs = np.zeros_like(v).astype(float)
        running_add = 0
        for t in reversed(range(0, discounted_ep_rs.shape[0])):
            running_add = running_add * self.gamma_value + v[t]
            discounted_ep_rs[t] = running_add
            discounted_ep_rs[discounted_ep_rs > 10] = 10
        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs