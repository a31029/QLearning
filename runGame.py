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
#     game = FlappyBird()
#     brain = Q(scale=7, explore=0)
#     try:
#         game.run(brain.run)
#     except:
#         brain.save()
#         print('\nsave Q in Matrix/Q.npy!!!')



# DQN算法 成功  直接把4维变量 作为 feature 扔进去 然后修改 reward 不死为1 死了为-10
# 但是Game 输出的是 不死为 score  死了为 -10
# 训练6分钟 已经破100了


if __name__ == "__main__":
    game = FlappyBird(is_speed_in_state = True,graph=True)
    brain = DQN(state_dim=4, explore=0.5, lr=1e-3, do_load = True, do_train=True,do_save=True)
    try:
        game.run(brain.run)
    except:
        brain.saveNet()
        print('\nsave Net parameters in Net/checkpoint!!!')






# Policy gradient 成功
# 修改方法是:
# 当speed 为负数时，即 小鸟正在向上冲刺，我们讲会对接下来几个state进行忽略，并不进行记录，当小鸟速度大于0时，即 开始下降时，我们会对后面的state 以及 action 进行记录。
# 即要求 小鸟不能无限制的一直往上飞，而必须满足当速度开始下降时 才能继续飞。
# 修改后的效果是 已经能够突破100了
# 并且训练的时候会将过去的几个 track 进行 记录 然后 训练的时候 会sample 出来1个进行训练。

# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True,graph=False)
#     brain = PG(lr = 1e-3,state_dim = 4,do_train=True,do_load=True,do_save = True)
#     try:
#         game.run(brain.run)
#     except:
#         brain.saveNet()



# AC算法 成功
# 进行了修改 查看一下 两部分其实没什么问题 但是因为 刚开始的时候 Qvalue 还不准，所以 对 actor的指引不好，导致actor 一直往错误的地方走，
# 当Q开始准了，actor 已经走到死胡同了 对 actor 的参数的导数为0了 训练不好了
# 所以现在是 先 随机action 先学 q 之后再 根据Q 调整actor 现在能跑到94 左右了

#  AC 或者 PG 算法 都有一个问题就是 训练比较慢 因为run一次 才能学一次 而 对于本游戏走到后面 run一次 可能要走10几步 比较慢

# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True,graph=False)
#     brain = AC(l1 = 1e-3,l2 = 1e-3,state_dim=4,do_train=True,do_load=True,do_save=True)
#     game.run(brain.run)
