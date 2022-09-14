import numpy as np
from epsilonGreedy import epsilonGreedy
import gym
from matplotlib import pyplot as plt
def SARSA(env:gym.Env, epsilon:float=0.7, gamma:float=0.9, episodes:int=10000, alpha : int=0.01) -> list:
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rwd = np.zeros(episodes)
    H = 10000
    e_1 = epsilon
    for i in range(episodes):
        print(f'Episode {i} Epsion {epsilon}')
        # Generating episode
        s = env.reset()
        a = epsilonGreedy(Q, epsilon, s)
        ts = 0
        while 1:
            
            
            s_1, r, done, _ = env.step(a)
            a_1 = epsilonGreedy(Q, epsilon, s_1)
            
            Q[s, a] = Q[s, a] +  alpha * (r + gamma * Q[s_1, a_1] - Q[s, a])
            
            s = s_1
            a = a_1
            
            rwd[i] += r
            ts += 1
            if done or r == -100 or ts > H:
                
                epsilon -= e_1 / (episodes - 10)
                break
    policy = np.argmax(Q, axis=1)
    return Q, policy, rwd


if __name__ == '__main__':  
    env = gym.make('CliffWalking')
    env.reset()
    Q, p, r = SARSA(env)
    plt.plot(r)
    plt.show()
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

        