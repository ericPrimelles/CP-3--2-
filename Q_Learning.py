import numpy as np
from epsilonGreedy import epsilonGreedy
from typing import Any

import gym
def QLearning(env: gym.Env, epsilon:float=0.7, gamma:float=0.9, episodes:int=10000, alpha : int=0.01) -> list:
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rwd = np.zeros(episodes)
    H = 10000
    e_1 = epsilon
    for i in range(episodes):
        # Generating episode
        s = env.reset()
        print(f'Episode {i} Epsilon: {epsilon}')
        ts = 0
        while 1:
           
            a = epsilonGreedy(Q, epsilon, s)
            s_1, r, done, _ = env.step(a)
            
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_1]) - Q[s, a])
            
            s = s_1
            
            rwd[i] += r
            ts += 1
            if done or r == -100 or ts > H:
                epsilon -= e_1 / episodes
                break
    policy = np.argmax(Q, axis=1)
    return Q, policy, rwd


import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    env = gym.make('CliffWalking')
    env.reset()
    Q, p, r = QLearning(env)
    plt.plot(r)
    plt.show()
    print(Q)
    s = env.reset()
    env.render()
    
    while 1:
        a = p[s]
        s_1, rwd, done, _ = env.step(a)
        env.render()
        s = s_1

        if done or rwd == -100:
            break
    
    env.close()
            
        
    