import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from collections import deque


class DQN():
    def __init__(self, explore=0, lr=1e-3, state_dim=5, do_load = False, do_train=False, do_save = False):
        
        self.explore = explore
        self.lr = lr
        self.state_dim = state_dim
        self.do_load = do_load
        self.do_train = do_train
        self.do_save = do_save

        self.gamma_value = 0.9

        self.train_count = 0

        self._get_session()
        self._build_net()
        self._build_memory()

        print('start DQN algorithm')

    def _get_session(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)


    def _build_net(self):
        self.input1 = tf.placeholder(tf.float32, [None, self.state_dim])
        self.w1_eval = tf.get_variable('w1_eval', dtype=tf.float32, shape=[self.state_dim, 32])
        self.b1_eval = tf.get_variable('b1_eval', dtype=tf.float32, shape=[32, ])
        self.w2_eval = tf.get_variable('w2_eval', dtype=tf.float32, shape=[32, 16])
        self.b2_eval = tf.get_variable('b2_eval', dtype=tf.float32, shape=[16, ])
        self.w3_eval = tf.get_variable('w3_eval', dtype=tf.float32, shape=[16, 2])
        self.b3_eval = tf.get_variable('b3_eval', dtype=tf.float32, shape=[2, ])

        self.v1_eval = tf.nn.leaky_relu(tf.matmul(self.input1, self.w1_eval) + self.b1_eval)
        self.v2_eval = tf.nn.leaky_relu(tf.matmul(self.v1_eval, self.w2_eval) + self.b2_eval)
        self.Q_eval = tf.matmul(self.v2_eval, self.w3_eval) + self.b3_eval

        self.input2 = tf.placeholder(tf.float32, [None, self.state_dim])
        self.w1_target = tf.get_variable('w1_target', dtype=tf.float32, shape=[self.state_dim, 32])
        self.b1_target = tf.get_variable('b1_target', dtype=tf.float32, shape=[32, ])
        self.w2_target = tf.get_variable('w2_target', dtype=tf.float32, shape=[32, 16])
        self.b2_target = tf.get_variable('b2_target', dtype=tf.float32, shape=[16, ])
        self.w3_target = tf.get_variable('w3_target', dtype=tf.float32, shape=[16, 2])
        self.b3_target = tf.get_variable('b3_target', dtype=tf.float32, shape=[2, ])

        self.v1_target = tf.nn.leaky_relu(tf.matmul(self.input2, self.w1_target) + self.b1_target)
        self.v2_target = tf.nn.leaky_relu(tf.matmul(self.v1_target, self.w2_target) + self.b2_target)
        self.Q_target = tf.matmul(self.v2_target, self.w3_target) + self.b3_target

        self.target = tf.placeholder(tf.float32,[None])
        self.act = tf.placeholder(tf.int32,[None])
        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(tf.multiply(self.Q_eval, tf.one_hot(self.act, 2)), reduction_indices=1) - self.target))
        
        self.current_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr,global_step = self.current_step,decay_steps=4000,decay_rate=0.95,staircase=True)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step = self.current_step)

        self.saver = tf.train.Saver()
        if len([str(x) for x in Path('Algorithm/DQN/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load:
            self.saver.restore(self.sess, "Algorithm/DQN/Net/model.ckpt")
            print('load net parameters!!!')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('init net parameters!!!')
        self._update()

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []

        self.memory = deque(self.loadFile('Algorithm/DQN/Memory/memory'),maxlen=5000)

    def _update(self):
        self.sess.run(tf.assign(self.w1_target, self.w1_eval))
        self.sess.run(tf.assign(self.b1_target, self.b1_eval))
        self.sess.run(tf.assign(self.w2_target, self.w2_eval))
        self.sess.run(tf.assign(self.b2_target, self.b2_eval))
        self.sess.run(tf.assign(self.w3_target, self.w3_eval))
        self.sess.run(tf.assign(self.b3_target, self.b3_eval))


    def _get_action(self,state):
        if np.random.uniform(0, 1) <= self.explore:
            action = np.random.choice([0,0,0,0,0,0,0,0,0,0,1])
        else:
            actions = self.sess.run(self.Q_eval, feed_dict={self.input1: np.array([state])})
            action = np.argmax(actions)
        return action


    def _remember_step(self,state,action,reward):
        if self.prev  is None:
            self.prev = np.hstack((state,action))
        else:
            self.memory_batch.append(np.hstack((self.prev, reward, state)))
            self.prev = np.hstack((state,action))


    def _remember_batch(self):
        self.memory.extend(self.memory_batch)
        self.memory_batch = []
        self.prev = None


    def run(self,now,dead,action):
        state = np.array([now[0] - now[2] ,now[1] - now[3] ,now[3], now[4]])
        action = self._get_action(state)
        if dead >= 0 :
            reward = 1
        else:
            reward = dead
        
        self._remember_step(state,action,reward)

        if dead <  0:
            self._remember_batch()
            self.train_count += 1

            if len(self.memory) > 2000:
                self.train()
                self.saveFile('Algorithm/DQN/Memory/memory',self.memory)
                self.saveNet()
                self.explore = self.explore * 0.3
                if self.explore <= 0.001:
                    self.explore = 0
            if self.train_count % 5 == 0 and len(self.memory) > 2000:
                self._update()
                self.plot()
        return action


    def train(self):
        if self.do_train:
            for i in range(100):
                length = min(len(self.memory),32)
                data_all = np.array(self.memory)
                data = data_all[np.random.choice(data_all.shape[0],length,replace=False)]

                q_ = self.sess.run(self.Q_target, feed_dict={self.input2: data[:,-1*self.state_dim:]})
                target = data[:,self.state_dim + 1] + (data[:,self.state_dim + 1] > 0) * self.gamma_value * np.max(q_, axis=1)
                act = data[:,self.state_dim].astype(int)
                _, loss_value, lr = self.sess.run([self.step, self.loss, self.learning_rate], feed_dict={self.input1: data[:, 0:self.state_dim], self.act: act, self.target: target})
            print('\t\tlearning rate: %s ;    loss: %s;    explore:%s.'%(lr,loss_value,self.explore))


    def saveNet(self):
        if self.do_save:
            self.saver.save(self.sess, "Algorithm/DQN/Net/model.ckpt")
            print('Net has been saved successfully')


    def saveFile(self,fileName,obj):
        if self.do_save:
            path = Path(fileName)
            with path.open('wb') as f:
                pickle.dump(obj,f)


    def loadFile(self,fileName):
        path = Path(fileName)
        if path.exists() and self.do_load :
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
        re = self.sess.run(self.Q_eval,feed_dict={self.input1:arr})
        img = ((re[:,0]-re[:,1])>0).astype(np.int).reshape(280,600).transpose(1,0)
        plt.imsave('FlappyBird/graph/z.png',img,cmap=plt.cm.gray)

