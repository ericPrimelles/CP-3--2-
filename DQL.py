import numpy as np
from epsilonGreedy import epsilonGreedy
import gym

'''

Double Q-Learning is an improvement made to vanilla Q-Learning. The goal of DQL is to reduce the over-optimistic behavior showed by Q-Learning
In this excersise you need to transform Q-Learning implementation into a Double Q-Learning implementation.

'''

def DQLearning(env: gym.Env, epsilon:float=0.7, gamma:float=0.9, episodes:int=20000, alpha : int=0.01) -> list:
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n)) # A second Q Table is maintained
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
            
            
           ''' Place your code here '''
                
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
    Q, p, r = DQLearning(env)
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
            
        
    
