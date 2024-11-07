import numpy as np
import random
import matplotlib.pyplot as plt


n_bandit = 2000         # number of bandit problems
k = 10                  # number of arms
n_pulls = 1000          # number of steps
col = ['b','r']

q_true = np.random.normal(0, 1, (n_bandit, k))  # true value

fig1=plt.figure().add_subplot(111)

methods = ['ucb', 'epsilon-greedy']


for c, m in enumerate(methods):
    avg_reward_pulls = []                       # average reward for per steps

    Q = np.zeros((n_bandit, k))                 # value estimate
    N = np.zeros((n_bandit, k))

    for t in range(1, 1 + n_pulls):
        # average reward for tasks in per step, use incremental formulation
        avg_reward_tasks = 0
      
        for i in range(n_bandit):
            if m == 'epsilon-greedy':
                if random.random() < epsilon:
                    j = np.random.randint(k)
                
                else:
                    max_val = np.max(Q[i])
                    max_indices = np.where(Q[i] == max_val)[0]
                    if len(max_indices) > 1:
                        j = random.choice(max_indices)
                    else:
                        j = max_indices[0]
            # ucb method
            else:
                if 0 in N[i]:
                    max_indices = np.where(N[i] == 0)[0]
                    if len(max_indices) > 1:
                        j = random.choice(max_indices)
                    else:
                        j = max_indices[0]
                else:
                    ucb_list = Q[i] + 2 * np.sqrt(np.log(t) / N[i])
                    max_val = np.max(ucb_list)
                    max_indices = np.where(ucb_list == max_val)[0]
                    if len(max_indices) > 1:
                        j = random.choice(max_indices)
                    else:
                        j = max_indices[0]
            
            temp_R = np.random.normal(q_true[i][j], 1)
            
            # sample-average action selection
            N[i][j] += 1
            Q[i][j] += (temp_R - Q[i][j]) / N[i][j]
            
            avg_reward_tasks += (temp_R - avg_reward_tasks) / (i + 1)
        
        avg_reward_pulls.append(avg_reward_tasks)
        
    # show results
    fig1.plot(range(1,n_pulls+1), avg_reward_pulls, col[c], linewidth = 1)
    


plt.rc('text',usetex=True)
fig1.title.set_text(r'$\varepsilon$-greedy : Average Reward Vs Steps for 10 arms')
fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')

fig1.legend((r"$\varepsilon$-greedy $\varepsilon$=0.1 ",\
			 r"UCB c = 2",),loc='best')

plt.show()

