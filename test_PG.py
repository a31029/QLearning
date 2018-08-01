import gym


import src.PolicyGradient as PG

env = gym.make('CartPole-v0')
env = env.unwrapped
state = env.reset()
brain = PG.Reinforcement(lr=0.02,state_dim= 4,load=False)
count =0
episode = 0


while(True):
    env.render()

    action = brain._get_action(state)
    state_,reward,done,info = env.step(action)
    assert reward == 1
    if done:
        reward = -10

    brain.memory_batch.append([state[0],state[1],state[2],state[3],action,reward])

    count += 1
    if done:
        brain.train()
        brain.memory = []
        print("Episode %s is completed. The total steps are %s" %
              (episode, count))
        count = 0
        episode += 1
        state = env.reset()
    else:
        state = state_

