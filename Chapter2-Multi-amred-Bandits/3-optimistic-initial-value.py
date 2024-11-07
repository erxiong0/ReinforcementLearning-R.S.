import numpy as np
import random
import matplotlib.pyplot as plt

n_bandit = 2000         # number of bandit problems
k = 10                  # number of arms
n_pulls = 1000          # number of steps

q_true = np.random.normal(0, 1, (n_bandit, k))  # true value
a_true_opt = np.argmax(q_true, 1)               # best action

fig1=plt.figure().add_subplot(111)


epsilons = [0, 0.1]
col = ['b', 'r']


for c, eps in enumerate(epsilons):
    optimal_actions = []                        # optimal_actions (%) for per steps

    Q = np.zeros((n_bandit, k))                 # value estimate
    if eps == 0:
        Q = np.ones((n_bandit, k)) * 5.0        

    for _ in range(n_pulls):
        
        # total optimal_actions for tasks in per step
        total_optimal_actions = 0
        for i in range(n_bandit):
            if random.random() < eps:
                j = np.random.randint(k)
            
            else:
                max_val = np.max(Q[i])
                max_indices = np.where(Q[i] == max_val)[0]
                if len(max_indices) > 1:
                    j = random.choice(max_indices)
                else:
                    j = max_indices[0]
                
            # if it is the optimal aciton
            if j == a_true_opt[i]:
                total_optimal_actions += 1
            
            temp_R = np.random.normal(q_true[i][j], 1)
            
            # constant step-size  
            Q[i][j] += (temp_R - Q[i][j]) * 0.1

        
        optimal_actions.append(total_optimal_actions * 100.0 / n_bandit)
    
    # show results
    fig1.plot(range(n_pulls), optimal_actions, col[c], linewidth = 1)


plt.rc('text',usetex=True)
fig1.title.set_text(r'$\varepsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
fig1.set_ylabel(r'$\%$ Optimal Action')
fig1.set_ylim(0, 100)
fig1.set_xlabel('Steps')
fig1.legend((r"$\varepsilon$=0, $Q_1$=5",\
			 r"$\varepsilon$=0.1, $Q_1$=0",),loc='best')



plt.show()
