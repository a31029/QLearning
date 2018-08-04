import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from pathlib import Path
import pickle
from collections import deque


class PG:
    def __init__(self,lr = 0.02,state_dim = 3,do_load = False,do_train = False,do_save = False):

        self.lr = lr
        self.state_dim = state_dim
        self.do_train = do_train
        self.do_load = do_load
        self.do_save = do_save

        self.gamma_value = 0.92

        self.t = None
        self.episode = 0

        self._get_session()
        self._build_memory()
        self._build_memory_pool()
        self._build_net()
        print('start Policy Gradient algorithm')


    def _get_session(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []

    def _build_memory_pool(self):
        self.memory_pool = deque([],maxlen=3)

    def _build_net(self):
        self.state = tf.placeholder(tf.float32,shape=[None,self.state_dim])

        self.first_layer = tf.layers.dense(self.state,32,activation=tf.nn.tanh)
        self.second_layer = tf.layers.dense(self.first_layer,16,activation=tf.nn.tanh)
        self.prob = tf.layers.dense(self.second_layer,2,activation=tf.nn.softmax)
    
        self.action = tf.placeholder(tf.int32,shape=[None])
        self.value = tf.placeholder(tf.float32,shape=[None])

        self.loss = tf.reduce_mean(tf.reduce_sum( tf.log( self.prob) * tf.one_hot(self.action, 2), axis=1) * self.value * -1)

        self.current_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr,global_step = self.current_step,decay_steps=400,decay_rate=0.9,staircase=True)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.current_step)

        self.saver = tf.train.Saver()
        if len([str(x) for x in Path('Algorithm/PG/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load :
            self.saver.restore(self.sess, "Algorithm/PG/Net/model.ckpt")
            print('load net parameters!!!')
        else:
            print('init net parameters!!!')
            self.sess.run(tf.global_variables_initializer())

    def _get_action(self,state):
        probs = self.sess.run(self.prob, feed_dict={self.state: np.array([state])})[0]
        # print(probs)
        return np.random.choice(2,p = probs)

    def _remember_step(self,state,action,reward):
        if self.prev  is None:
            self.prev = np.hstack((state,action))
        else:
            self.memory_batch.append(np.hstack((self.prev, reward)))
            self.prev = np.hstack((state,action))

    def _remember(self):
        self.memory_pool.append(self.memory_batch)

    def run(self, now, dead, action):
        if self.t is None:
            self.t = 0
        else:
            self.t += 1

        state = np.array([now[0] - now[2] ,now[1] - now[3] ,now[3], now[4]])
        action = self._get_action(state)
        
        if dead >= 0 :
            reward = 1
        else:
            reward = dead

        self._remember_step(state, action, reward)

        if dead < 0:
            self._remember()
            print("Episode %s is completed. The total steps are %s" % (self.episode,self.t))
            self.train()
            self.memory_batch = []
            self.prev = None

            self.episode += 1
            self.t = None

            if self.episode % 10 == 0 :
                self.saveNet()
                self.plot()
        return action

    def train(self):
        if self.do_train:
            for i in range(1):
                m = np.array(self.memory_pool[np.random.choice(len(self.memory_pool))])
                s = m[:,0:self.state_dim]
                a = m[:, self.state_dim]
                v = self._discount_and_norm_rewards(m[:, self.state_dim+1])
                _, loss_value, lr, prob_value = self.sess.run([self.step, self.loss, self.learning_rate,self.prob], feed_dict={self.state: s, self.action: a, self.value: v})
                print("lr:%s,    loss:%s."%(lr,loss_value))


    def _discount_and_norm_rewards(self,v):
        discounted_ep_rs = np.zeros_like(v).astype(float)
        running_add = 0
        for t in reversed(range(0, discounted_ep_rs.shape[0])):
            running_add = running_add * self.gamma_value + v[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs

    def saveNet(self):
        if self.do_save:
            self.saver.save(self.sess, "Algorithm/PG/Net/model.ckpt")
            print("net has been saved successfully")

    def plot(self):

        xx,yy = np.meshgrid(np.arange(600),np.arange(280))
        x = xx.flatten() - 228
        y = yy.flatten() - 380
        yd = np.ones_like(x) * 360
        sp = np.ones_like(y) * 4.5
        arr = np.vstack((x,y,yd,sp)).transpose(1,0)
        re = self.sess.run(self.prob,feed_dict={self.state:arr})
        img = ((re[:,0]-re[:,1])>0).astype(np.int).reshape(280,600).transpose(1,0)
        plt.imsave('FlappyBird/graph/z.png',img,cmap=plt.cm.gray)

# brain = PG(lr = 1e-3,state_dim = 4,do_train=True,do_load=False,do_save = True)
# brain.plot()