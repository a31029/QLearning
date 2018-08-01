import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from pathlib import Path
import pickle

from FlappyBird.Config import Parameters
from FlappyBird.Util import Index


class Reinforcement:
    def __init__(self,scale = 1,lr = 0.02,state_dim = 3,load = True,do_train = False):
        self.index = Index(scale, Parameters.state_x_min, Parameters.state_x_max,Parameters.state_y_min, Parameters.state_y_max)

        self.size = self.index._shape

        self.lr = lr
        self.do_train = do_train
        self.gamma = 0.92

        self.state_dim = state_dim

        tf.reset_default_graph()

        self.memory_batch = []

        self.prev = None
        self.load = load
        
        self._get_session()
        self._build_net()

        self.episode = 0

        # self.trainFlg = self.trainFlg = 30000 + \
        #     len([str(p) for p in Path('graph').iterdir() if p.match('*.png')]) - 1

        self.t = None


    def _get_session(self):
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

    def _build_net(self):
        self.state = tf.placeholder(tf.float32,shape=[None,self.state_dim])

        self.first_layer = tf.layers.dense(self.state,32,activation=tf.nn.tanh,use_bias=True,bias_initializer=tf.random_uniform_initializer(-3,3))
        self.second_layer = tf.layers.dense(self.first_layer,16,activation=tf.nn.tanh,use_bias=True,bias_initializer=tf.random_uniform_initializer(-3,3))
        self.prob = tf.layers.dense(self.second_layer,2,activation=tf.nn.softmax)
    
        self.action = tf.placeholder(tf.int32,shape=[None])
        self.value = tf.placeholder(tf.float32,shape=[None])

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.log(self.prob) * tf.one_hot(self.action, 2), axis=1) * self.value)

        self.current_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(-1 * self.lr,global_step = self.current_step,decay_steps=10000,decay_rate=0.95,staircase=True)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.current_step)

        self.saver = tf.train.Saver()

        if len([str(x) for x in Path('Algorithm/PG/Net').iterdir() if x.match('model.ckpt*')]) != 0 and self.load :
            self.saver.restore(self.sess, "Algorithm/PG/Net/model.ckpt")
            print('load parameters--------------------------!')
        else:
            print('init parameters--------------------------!')
            self.sess.run(tf.global_variables_initializer())


    def _get_action(self,state):
        probs = self.sess.run(self.prob, feed_dict={self.state: np.array([state])})
        return np.random.choice(2,p = probs[0])
        # return np.argmax(probs)

    def _remember(self,state,action,reward):
        if self.prev is None:
            self.prev = np.hstack((state,action))
        else:
            self.prev = np.hstack((self.prev, reward))
            self.memory_batch.append(self.prev)
            self.prev = np.hstack((state, action))

    def run(self, now, dead, action):
        if self.t is None:
            self.t = 0

        state = now

        action = self._get_action(state)

        if dead < 0:
            reward = dead
        else:
            reward = 1

        self._remember(state, action, reward)
        self.t += 1

        if dead < 0 and self.do_train:
            print("Episode %s is completed. The total steps are %s" % (self.episode,self.t))
            self.train()
            self.memory_batch = []
            self.prev = None

            self.episode += 1
            self.t = None
            if self.episode % 10 == 0:
                self.saveNet()
                # self.plot()
        return action

    def train(self):
        m = np.array(self.memory_batch)
        s = m[:,0:self.state_dim]
        a = m[:, self.state_dim]
        v = self._discount_and_norm_rewards(m[:, self.state_dim+1])
        _, loss_value, lr, prob_value = self.sess.run([self.step, self.loss, self.learning_rate,self.prob], feed_dict={self.state: s, self.action: a, self.value: v})
        print(">>>>>>>>>>>>>lr:%s<<<<<<<<<<<<<<%s"%(lr,loss_value))
        # print(prob_value[np.random.choice(prob_value.shape[0],5)])
        # print(v)


    def _discount_and_norm_rewards(self,v):
        discounted_ep_rs = np.zeros_like(v).astype(float)
        running_add = 0
        for t in reversed(range(0, discounted_ep_rs.shape[0])):
            running_add = running_add * self.gamma + v[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print(discounted_ep_rs)
        return discounted_ep_rs

    # def plot(self):
    #     xx, yy = np.meshgrid(np.linspace(0, self.size[0]-1, self.size[0]).astype(int),
    #                          np.linspace(0, self.size[1]-1, self.size[1]).astype(int))

    #     x = np.vstack((xx.flatten('F'), yy.flatten('F'))).transpose(1, 0)
    #     output = self.sess.run(self.prob, feed_dict={self.state: x})

    #     # diff = np.zeros((output.shape[0]))
    #     # diff[output[:, 1] < output[:, 0]] = 1
    #     # diff[output[:, 1] > output[:, 0]] = -1
    #     diff = output[:,0] - output[:,1]
    #     diff = diff.reshape(self.size).transpose(1, 0)

    #     diff = misc.imresize(diff, (self.index._y_max-self.index._y_min, self.index._x_max-self.index._x_min))
    #     plt.imsave("graph/show%s.png" % self.trainFlg, diff, cmap=plt.cm.gray)
    #     self.trainFlg += 1


    def saveNet(self):
        self.saver.save(self.sess, "Algorithm/PG/Net/model.ckpt")
        print("net saved success")

