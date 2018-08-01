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

        self.gamma_reward = 0.92
        self.gamma_value = 0.9

        self.train_count = 0
        # self.llist = []

        self._get_session()
        self._build_net()
        self._build_memory()

    def _get_session(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

    def _build_net(self):

        self.input1 = tf.placeholder(tf.float32, [None, self.state_dim])
        self.w1_eval = tf.get_variable('w1_eval', dtype=tf.float32, shape=[self.state_dim, 32])
        self.b1_eval = tf.get_variable('b1_eval', dtype=tf.float32, shape=[32, ])
        self.w2_eval = tf.get_variable('w2_eval', dtype=tf.float32, shape=[32, 32])
        self.b2_eval = tf.get_variable('b2_eval', dtype=tf.float32, shape=[32, ])
        self.w3_eval = tf.get_variable('w3_eval', dtype=tf.float32, shape=[32, 2])
        self.b3_eval = tf.get_variable('b3_eval', dtype=tf.float32, shape=[2, ])

        self.v1_eval = tf.nn.leaky_relu(tf.matmul(self.input1, self.w1_eval) + self.b1_eval)
        self.v2_eval = tf.nn.leaky_relu(tf.matmul(self.v1_eval, self.w2_eval) + self.b2_eval)
        self.Q_eval = tf.matmul(self.v2_eval, self.w3_eval) + self.b3_eval
    
        self.input2 = tf.placeholder(tf.float32, [None, self.state_dim])
        self.w1_target = tf.get_variable('w1_target', dtype=tf.float32, shape=[self.state_dim, 32])
        self.b1_target = tf.get_variable('b1_target', dtype=tf.float32, shape=[32, ])
        self.w2_target = tf.get_variable('w2_target', dtype=tf.float32, shape=[32, 32])
        self.b2_target = tf.get_variable('b2_target', dtype=tf.float32, shape=[32, ])
        self.w3_target = tf.get_variable('w3_target', dtype=tf.float32, shape=[32, 2])
        self.b3_target = tf.get_variable('b3_target', dtype=tf.float32, shape=[2, ])

        self.v1_target = tf.nn.leaky_relu(tf.matmul(self.input2, self.w1_target) + self.b1_target)
        self.v2_target = tf.nn.leaky_relu(tf.matmul(self.v1_target, self.w2_target) + self.b2_target)
        self.Q_target = tf.matmul(self.v2_target, self.w3_target) + self.b3_target

        self.target = tf.placeholder(tf.float32,[None])
        self.act = tf.placeholder(tf.int32,[None])
        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(tf.multiply(self.Q_eval, tf.one_hot(self.act, 2)), reduction_indices=1) - self.target))
        
        self.current_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr,global_step = self.current_step,decay_steps=5000,decay_rate=0.95,staircase=True)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step = self.current_step)

        self.saver = tf.train.Saver()
        if len([str(x) for x in Path('Algorithm/DQN/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.do_load:
            self.saver.restore(self.sess, "Algorithm/DQN/Net/model.ckpt")
        else:
            self.sess.run(tf.global_variables_initializer())

        self._update()

    def _build_memory(self):
        self.prev = None
        self.memory_batch = []

        self.memory = deque(self.loadFile('Algorithm/DQN/Memory/memory'),maxlen=10000)
        # self.memory_pos = deque(self.loadFile('Algorithm/DQN/Memory/memory_pos'), maxlen=10000)
        # self.memory_neg = deque(self.loadFile('Algorithm/DQN/Memory/memory_neg'), maxlen=10000)

    def _update(self):
        self.sess.run(tf.assign(self.w1_target, self.w1_eval))
        self.sess.run(tf.assign(self.b1_target, self.b1_eval))
        self.sess.run(tf.assign(self.w2_target, self.w2_eval))
        self.sess.run(tf.assign(self.b2_target, self.b2_eval))
        self.sess.run(tf.assign(self.w3_target, self.w3_eval))
        self.sess.run(tf.assign(self.b3_target, self.b3_eval))

    def _get_action(self,state):
        if np.random.uniform(0, 1) <= self.explore:
            # print("explore!!!")
            action = np.random.choice([0,0,0,0,0,0,0,0,0,1])
        else:
            actions = self.sess.run(self.Q_eval, feed_dict={self.input1: np.array([state])})
            # print(actions)
            action = np.argmax(actions)
        return action



    def run(self,now,dead,action):

        state = now

        action = self._get_action(state)

        if dead >= 0 :
            reward = 1
        else:
            reward = dead

        if self.prev  is None:
            self.prev = np.hstack((state,action))
        else:
            self.memory_batch.append(np.hstack((self.prev, reward, state)))
            self.prev = np.hstack((state,action))

        if dead <  0:
            self.train_count += 1
            self.prev = None

            tmp = np.array(self.memory_batch)
            tmp[:,self.state_dim+1] = self._discount_and_norm_rewards(tmp[:,self.state_dim+1])
            self.memory.extend(tmp.tolist())

            if self.train_count % 20 == 0  and self.do_train:
                self.train()
                self.saveFile('Algorithm/DQN/Memory/memory',self.memory)
                self.saveNet()
                self.explore = self.explore * 0.3
                if self.explore <= 0.001:
                    self.explore = 0
            if self.train_count % 400 == 0 and self.do_train:
                self._update()
        return action



    def train(self):
        for i in range(int(500 * self.explore)+120):

            length = min(len(self.memory),24)
            data_all = np.array(self.memory)
            data = data_all[np.random.choice(data_all.shape[0],length,replace=False)]
    
            q_ = self.sess.run(self.Q_target, feed_dict={self.input2: data[:,-1*self.state_dim:]})
            act = data[:,self.state_dim].astype(int)
            indR_neg = (data[:, self.state_dim + 1] < 0).astype(int)
            target = data[:,self.state_dim + 1] + (1 - indR_neg) * self.gamma_value *np.max(q_, axis=1)

            _, loss_value, lr = self.sess.run([self.step, self.loss, self.learning_rate], feed_dict={self.input1: data[:, 0:self.state_dim], self.act: act, self.target: target})

            # self.llist.append(loss_value)
        print('>>>>>>>>>>>>>>>>> lr: %s <<<<<<<<<<<<<<<<<<<<<<<'%lr)
        print(self.explore,loss_value)

    def _discount_and_norm_rewards(self,v):
        discounted_ep_rs = np.zeros_like(v).astype(float)
        running_add = 0
        for t in reversed(range(0, discounted_ep_rs.shape[0])):
            running_add = running_add * self.gamma_reward + v[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs


    def saveNet(self):
        if self.do_save:
            self.saver.save(self.sess, "Algorithm/DQN/Net/model.ckpt")

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

