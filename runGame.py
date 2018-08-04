import numpy as np
import tensorflow as tf

import collections
import pygame
import os
import random
import gym

from FlappyBird.Game import FlappyBird
from Algorithm.QLearning.QL import Q
from Algorithm.DQN.DQN import DQN
from Algorithm.PG.PolicyGradient import PG
from Algorithm.AC.AC import AC

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# random.seed(1)
# tf.set_random_seed(1)
# np.random.seed(1)

# 运行游戏，无算法

# if __name__ == "__main__":
#     game = FlappyBird()
#     game.run()



# Q-table 算法


# if __name__ == "__main__":
#     game = FlappyBird(interval = 1)
#     brain = Q(scale=7, explore=0)
#     try:
#         game.run(brain.run)
#     except:
#         brain.save()
#         print('\nsave Q in Matrix/Q.npy!!!')



# DQN算法 成功  直接把4维变量 作为 feature 扔进去 然后修改 reward 不死为1 死了为-10
# 但是Game 输出的是 不死为 score  死了为 -10
# 训练6分钟 已经破100了


# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True,graph=True)
#     brain = DQN(state_dim=4, explore=0.5, lr=1e-3, do_load = True, do_train=True,do_save=True)
#     try:
#         game.run(brain.run)
#     except:
#         brain.saveNet()
#         print('\nsave Net parameters in Net/checkpoint!!!')




# Policy gradient 效果不太好，需要长时间的训练 并且训练并不稳定 通过不断的修改 lr 才能到现在的成都
# 最多可以到 30  但是 基于 policy 的方法 始终没办法获得更好的效果 可能是我对 policy based的 还是存在一些理解的问题


# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True,graph=True,interval = 3)
#     brain = PG(lr = 1e-3,state_dim = 4,do_train=True,do_load=True,do_save = True)
#     try:
#         game.run(brain.run)
#     except:
#         brain.saveNet()



# AC算法 成功
# 进行了修改 查看一下 两部分其实没什么问题 但是因为 刚开始的时候 Qvalue 还不准，所以 对 actor的指引不好，导致actor 一直往错误的地方走，
# 当Q开始准了，actor 已经走到死胡同了 对 actor 的参数的导数为0了 训练不好了
# 所以现在是 先 随机action 先学 q 之后再 根据Q 调整actor 现在能跑到15左右了

#  AC 或者 PG 算法 都有一个问题就是 训练比较慢 因为run一次 才能学一次 而 对于本游戏走到后面 run一次 可能要走10几步 比较慢

if __name__ == "__main__":
    game = FlappyBird(is_speed_in_state = True,graph=True,interval = 4)
    brain = AC(l1 = 1e-3,l2 = 1e-3,state_dim=4,do_train=True,do_load=True,do_save=True)
    try:
        game.run(brain.run)
    except:
        brain.saveNet()

