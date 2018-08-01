#用写好的DDPG 运行 bird 不成功 所以改run cartpole 试一下
#结果发现 还是可以很好的运行的 跟 policy gradient的情形 类似
#都是 Policy 和 DDPG 都可以用在 cartpole 上，但是用bird 就不行
#我也不知道为啥

#运行 test 和运行 flappy bird  只需要改两处 1.memorize  2.网络的 输入dim

import gym

import numpy as np
from Algorithm.AC.AC import AC

env = gym.make('CartPole-v0')
env = env.unwrapped

state = env.reset()
brain = AC(load=False, state_dim=4)

count = 0
episode = 0


while(True):
    env.render()
    ch_state = np.array([state])
    action = brain.get_action(ch_state)
    state_, reward, done, info = env.step(action)
    assert reward == 1
    if done:
        reward = -10
    mm = np.hstack((state,action,reward,state_))
    brain.memory.append(mm)
    count += 1
    if done:
        brain.train()

        print("Episode %s is completed. The total steps are %s" %
              (episode, count))
        count = 0
        episode += 1
        state = env.reset()
    else:
        state = state_
