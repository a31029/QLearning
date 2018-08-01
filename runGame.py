

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
from Algorithm.PG.PolicyGradient import Reinforcement
from Algorithm.AC.AC import AC

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# random.seed(1)
# tf.set_random_seed(1)
# np.random.seed(1)


# # 运行游戏，无算法
# if __name__ == "__main__":
#     game = FlappyBird()
#     while(True):
#         game.run()


# # Q-table 算法
# if __name__ == "__main__":
#     game = FlappyBird()
#     brain = Q(scale=7, explore=0)
#     try:
#         game.run(brain.run)
#     except:
#         brain.save()
#         print('\nsave Q in Matrix/Q.npy!!!')



# DQN算法 成功  直接把5维变量 作为 feature 扔进去 然后修改 reward 不死为1 死了为-10
# 但是Game 输出的是 不死为 score  死了为 -10
# 同样  对 reward 进行了特征工程 同样用PG里面的 对reward 进行了一个 decay 会进行传递 而不是直接用 这个1 -10  因为到了最后碰壁的时候
# 可能已经进入死胡同了 不管怎么走 都是死 应该要把这个 反馈信息传递给过去的几个 动作 。所以 这个传递就是通过 _discount_and_norm_rewards() 函数实现的
# 目前已经到了 654了 感觉会一直往下走

if __name__ == "__main__":
    game = FlappyBird(is_speed_in_state = True)
    brain = DQN(state_dim=4, explore=0, lr=1e-3, do_load = True, do_train=True,do_save=True)
    game.run(brain.run)


# Policy gradient  算法成功  直接把5维变量 作为 feature 进行了输入 然后修改reward 不死为1  死了为-10 
# 看来啊  feature特征 、reward 修改、网络capacity 的确是 提高算法稳定性的重要工具
# 目前最高已经可以到100多了 但是训练的很慢 因为 基本上得 跑20多个柱子才能训练一次 但是 DQN 跑一次 可能从 memory中 拿出来 好几次的经验去训练
# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True)
#     brain = Reinforcement(scale = 1,lr = 1e-3,state_dim = 5,do_train=True)
#     game.run(brain.run)


# AC算法 成功
# 进行了修改 查看一下 两部分其实没什么问题 但是因为 刚开始的时候 Qvalue 还不准，所以 对 actor的指引不好，导致actor 一直忘错误的地方走，
# 当Q开始准了，actor 已经走到死胡同了 对 actor 的参数的导数为0了 训练不好了
# 所以现在是 先 随机action 先学 q 之后再 根据Q 调整actor 现在能跑到15左右了

# AC 或者 PG 算法 都有一个问题就是 训练比较慢 因为run一次 才能学一次 而 对于本游戏走到后面 run一次 可能要走10几步 比较慢

# if __name__ == "__main__":
#     game = FlappyBird(is_speed_in_state = True)
#     brain = AC(scale=1,load=True, state_dim=5)
#     while(True):
#         game.run(brain.run)

