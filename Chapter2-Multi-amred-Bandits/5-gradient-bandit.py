import numpy as np
import random
import matplotlib.pyplot as plt


n_bandit = 2000         # number of bandit problems
k = 10                  # number of arms
n_pulls = 1000          # number of steps

q_true = np.random.normal(4, 1, (n_bandit, k))  # true value
a_true_opt = np.argmax(q_true, 1)

fig1=plt.figure().add_subplot(111)

methods = [0.1, 0.4, 0.1, 0.4]      # alpha
col = ['b', '#ADD8E6','brown','#DEB887']


for c, alpha in enumerate(methods):
    optimal_actions = []                        # optimal_actions (%) for per steps

    R = np.zeros(n_bandit)                 # baseline for per bandit task, using incremental formulation
    H = np.zeros((n_bandit, k))            
    
    for t in range(1, n_pulls + 1):
        # total optimal_actions for tasks in per step
        total_optimal_actions = 0
        
        for i in range(n_bandit):
            exp = np.exp(H[i] - max(H[i]))  # - max(H[i]) for numerical stability
            prob = exp / sum(exp)    
            j = np.random.choice(range(k), 1, p = prob)
                
            # if it is the optimal aciton
            if j == a_true_opt[i]:
                total_optimal_actions += 1
            
            temp_R = np.random.normal(q_true[i][j], 1)

            # update H[i]
            if c > 1:       
                R[i] = 0        # without baseline
            
            if t > 1:
                for a in range(k):
                    if a == j:
                        H[i][a] += alpha * (temp_R - R[i]) * (1 - prob[a])
                    else:
                        H[i][a] -= alpha * (temp_R - R[i]) * prob[a]

            # update H[i]
            if c > 1:       
                R[i] = 0        # without baseline
            else:
                R[i] += (temp_R - R[i]) / t

        
        optimal_actions.append(total_optimal_actions * 100.0 / n_bandit)
    
    # show results
    fig1.plot(range(n_pulls), optimal_actions, col[c], linewidth = 1)


plt.rc('text',usetex=True)
fig1.title.set_text(r'Gradient bandit algorithm : $\%$ Optimal Action Vs Steps for 10 arms')
fig1.set_ylabel(r'$\%$ Optimal Action')
fig1.set_ylim(0, 100)
fig1.set_xlabel('Steps')
fig1.legend((r"$\alpha$=0.1, with baseline",\
			r"$\alpha$=0.4, with baseline",\
            r"$\alpha$ = 0.1, without baseline",\
            r"$\alpha$ = 0.4, without baseline",),loc='best')

fig1.spines['right'].set_visible(False)
fig1.spines['top'].set_visible(False)
fig1.tick_params(direction='in', which='both', axis='both')

plt.show()


