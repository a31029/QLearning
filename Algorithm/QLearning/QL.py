import numpy as np
import random
from pathlib import Path

from FlappyBird.Util import Index
from FlappyBird.Config import Parameters



class Q:
    def __init__(self,scale=7,explore = 0.01,do_load = True):
        self.index = Index(scale, Parameters.state_x_min,Parameters.state_x_max, Parameters.state_y_min, Parameters.state_y_max)
        self.size = self.index._shape
        if do_load:
            self.Q = self.load()
        if self.Q is None or self.Q.shape[0] != self.size[0] or self.Q.shape[1] != self.size[1]:
            print("load Q matrix failed, creating a new one...")
            self.Q = np.zeros((self.size[0], self.size[1], 2))
        self.explore = explore
        self.prev = None
        self.gamma = 0.9
        self.lr = 0.7
        self.episode = 0
        print('start Q-table algorithm')

    
    def run(self, now, dead, action):

        state = self.index.trans2d(now)
        actions = self.Q[state[0],state[1]]
        
        if np.random.uniform(0, 1) < self.explore:
            action = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            action = np.argmax(actions)

        reward = dead

        if self.prev is None :
            self.prev = [state[0],state[1], action]
        else:
            self.Q[self.prev[0],self.prev[1], self.prev[2]] += self.lr * (reward + self.gamma * (np.max(self.Q[state[0], state[1]])) - self.Q[self.prev[0], self.prev[1], self.prev[2]])
            if reward < 0 :
                self.prev = None
                self.Q[state[0],state[1]] = reward
            else:
                self.prev = [state[0],state[1], action]
        return action

    def save(self):
        np.save('Algorithm/QLearning/Matrix/Q.npy',self.Q)
        print('Q table has been saved successfully')

    def load(self):
        p = Path('Algorithm/QLearning/Matrix/Q.npy')
        if p.exists():
            return np.load('Algorithm/QLearning/Matrix/Q.npy')
        else:
            return None
