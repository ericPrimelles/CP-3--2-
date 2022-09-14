import numpy as np

def epsilonGreedy(Q : np.float32, epsilon : float, state : int) -> int:
    
    if np.random.uniform(0, 1, 1) < epsilon:
        
        # Exploring
        
        return np.random.randint(0, len(Q[0]), 1)[0]
    
    # Greedy
    
    return np.argmax(Q[state])


if __name__ == '__main__':
    
    
    for i in range(100):
       
        print(epsilonGreedy(np.random.random((10,4)), 0.1, np.random.randint(0, 10, 1)[0]))
    
    