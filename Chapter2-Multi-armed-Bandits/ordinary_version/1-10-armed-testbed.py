import numpy as np
from typing import List
import random
import matplotlib.pyplot as plt
import subprocess

n_bandit = 2000     # number of bandit problems
k = 10              # number of arms
n_pulls = 1000      # number of steps

epsilons = [0, 0.01, 0.1, 0.2, 1]   
col=['r','g','k','b','y'] 

# true value
q_true = np.random.normal(0, 1, (n_bandit, k))
# best action
a_true_opt = np.argmax(q_true, 1)

fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for c, eps in enumerate(epsilons):
    # average reward for per steps
    avg_reward_pulls = []
    # optimal_actions (%) for per steps
    optimal_actions = []

    # count action
    N = np.zeros((n_bandit, k))
    # value estimate
    Q = np.zeros((n_bandit, k))

    for _ in range(n_pulls):
        # average reward for tasks in per step, use incremental formulation
        avg_reward_tasks = 0
        # total optimal_actions for tasks in per step
        total_optimal_actions = 0
        for i in range(n_bandit):
            # epsilon-greedy method
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
            
            # reward
            temp_R = np.random.normal(q_true[i][j], 1)
            # sample-average action selection
            N[i][j] += 1
            Q[i][j] += (temp_R - Q[i][j]) / N[i][j]

            avg_reward_tasks += (temp_R - avg_reward_tasks) / (i + 1)
        
        avg_reward_pulls.append(avg_reward_tasks)
        optimal_actions.append(total_optimal_actions * 100.0 / n_bandit)
    
    # show results
    fig1.plot(range(n_pulls), avg_reward_pulls, col[c], linewidth = 1)
    fig2.plot(range(n_pulls), optimal_actions, col[c], linewidth = 1)


# check latex package
def is_latex_installed():
    try:
        subprocess.run(['latex', '--version'], check = True, stdout = subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if is_latex_installed():
    plt.rc('text', usetex = True)

    fig1.title.set_text(r'$\varepsilon$-greedy : Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.legend((r"$\epsilon=$"+str(epsilons[0]),\
                r"$\epsilon=$"+str(epsilons[1]),\
                    r"$\epsilon=$"+str(epsilons[2]),\
                        r"$\epsilon=$"+str(epsilons[3]),\
                            r"$\epsilon=$"+str(epsilons[4])),loc='best')
    fig2.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 10 arms')
    fig2.set_ylabel(r'$\%$ Optimal Action')
    fig2.set_xlabel('Steps')
    fig2.set_ylim(0,100)
    fig2.legend((r"$\epsilon=$"+str(epsilons[0]),\
                r"$\epsilon=$"+str(epsilons[1]),\
                    r"$\epsilon=$"+str(epsilons[2]),\
                        r"$\epsilon=$"+str(epsilons[3]),\
                            r"$\epsilon=$"+str(epsilons[4])),loc='best')


else:
    fig1.title.set_text('Epsilon-greedy: Average Reward Vs Steps for 10 arms')
    fig1.set_ylabel('Average Reward')
    fig1.set_xlabel('Steps')
    fig1.legend(('epsilon = ' + str(epsilons[0]), \
                'epsilon = ' + str(epsilons[1]), \
                'epsilon = ' + str(epsilons[2]), \
                'epsilon = ' + str(epsilons[3]), \
                'epsilon = ' + str(epsilons[4])), loc = 'best')
    
    fig2.title.set_text(r'Epsilon-greedy: % Optimal Action Vs Steps for 10 arms')
    fig2.set_ylabel('% Optimal Action')
    fig2.set_xlabel('Steps')
    fig2.set_ylim(0, 100)
    fig2.legend(('epsilon = ' + str(epsilons[0]), \
                'epsilon = ' + str(epsilons[1]), \
                'epsilon = ' + str(epsilons[2]), \
                'epsilon = ' + str(epsilons[3]), \
                'epsilon = ' + str(epsilons[4])), loc = 'best')

plt.show()