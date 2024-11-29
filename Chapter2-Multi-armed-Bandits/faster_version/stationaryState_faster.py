import torch
import time
import os
import matplotlib.pyplot as plt
import subprocess


current_dir = os.path.abspath(os.path.dirname(__file__))

def epsilon_greedy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_bandit = 2000     # number of bandit problems
    k = 10              # number of arms of each bandit problem
    n_pulls = 1000      # number of steps

    epsilon_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]    # epsilon-greedy method
    q_true = torch.normal(0, 1, (n_bandit, k), device = device, requires_grad = False)

    eps_res_total = []      # save total resutls

    for eps in epsilon_values:
        # value estimate
        Q = torch.zeros(n_bandit, k, device = device, requires_grad = False)
        N = torch.zeros(n_bandit, k, device = Q.device, requires_grad = False)

        # average reward per epsilon
        avg_eps_res = 0
        for pull in range(1, n_pulls + 1):
            # epsilon greedy method
            epsilon_mask = torch.rand(n_bandit, device = device) < eps
            random_actions = torch.randint(0, k, (n_bandit,), device = device)

            chosen_actions = torch.where(epsilon_mask.unsqueeze(1), random_actions.unsqueeze(1), torch.argmax(Q, dim = 1, keepdim = True))
            temp_R = torch.zeros(n_bandit, k, device = Q.device)

            bandit_idx = torch.arange(n_bandit, device = device)
            chosen_idx = chosen_actions.squeeze().to(device)
            temp_R[bandit_idx, chosen_idx] = torch.randn(n_bandit, device = q_true.device) * 1 + \
            q_true[bandit_idx, chosen_idx]
            N[bandit_idx, chosen_idx] += 1
            Q[bandit_idx, chosen_idx] += \
            (temp_R[bandit_idx, chosen_idx] - Q[bandit_idx, chosen_idx]) / N[bandit_idx, chosen_idx]

            avg_eps_res += (torch.mean(temp_R[bandit_idx, chosen_idx]).item() - avg_eps_res) / pull
        
        eps_res_total.append(avg_eps_res)
    
    eps_res_total = torch.tensor(eps_res_total)
    epsilon_values = torch.tensor(epsilon_values)

    torch.save(eps_res_total, f'{current_dir}/torch_results/epsilon-weighted.pth')
    torch.save(epsilon_values, f'{current_dir}/torch_results/epsilon-x.pth')


def ucb():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_bandit = 2000     # number of bandit problem
    k = 10              # number of each bandit problem
    n_pulls = 1000      # steps

    q_true = torch.normal(0, 1, (n_bandit, k), device = device, requires_grad = False)

    C = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    ucb_res_total = []  # save total results

    for c in C:
        Q = torch.zeros((n_bandit, k), device = device, requires_grad = False)   # initial value estimate
        N = torch.zeros((n_bandit, k), device = Q.device, requires_grad = False)   # action value

        avg_ucb_res = 0

        for n_pull in range(1, n_pulls + 1):
            log_n_pull = torch.log(torch.tensor([n_pull], device = Q.device)).unsqueeze(-1).expand_as(N)
            mask = (N > 0)
            ucb_Q = torch.full(Q.shape, float('inf'), device = Q.device)
            ucb_Q[mask] = Q[mask] + c * torch.sqrt(log_n_pull[mask] / N[mask])
            chosen_action = torch.argmax(ucb_Q, dim = 1, keepdim = True)

            temp_R = torch.zeros(n_bandit, k, device = q_true.device)

            bandit_idx = torch.arange(n_bandit, device = device)
            chosen_idx = chosen_action.squeeze().to(device)

            temp_R[bandit_idx, chosen_idx] = torch.randn(n_bandit, device = q_true.device) * 1 + \
            q_true[bandit_idx, chosen_idx]
            N[bandit_idx, chosen_idx] += 1
            Q[bandit_idx, chosen_idx] += \
            (temp_R[bandit_idx, chosen_idx] - Q[bandit_idx, chosen_idx]) / N[bandit_idx, chosen_idx]

            avg_ucb_res += (torch.mean(temp_R[bandit_idx, chosen_idx]).item() - avg_ucb_res) / n_pull
        
        ucb_res_total.append(avg_ucb_res)
    
    ucb_res_total = torch.tensor(ucb_res_total)
    C = torch.tensor(C)
    torch.save(ucb_res_total, f'{current_dir}/torch_results/ucb-weighted.pth')
    torch.save(C, f'{current_dir}/torch_results/ucb-x.pth')

def gradient_bandit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_bandit = 2000     # number of bandit
    k = 10              # number of arms
    n_pulls = 1000      # steps

    q_true = torch.normal(0, 1, (n_bandit, k), device = device, requires_grad = False)

    ALPHA = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]

    g_res_total = []

    for alpha in ALPHA:
        H = torch.zeros((n_bandit, k), device = q_true.device)

        avg_alpha_res = 0       # average reward of each alpha
        Q = torch.zeros(n_bandit, device = q_true.device)

        for n_pull in range(1, n_pulls + 1):
            prob = torch.softmax(H - torch.max(H, dim = -1, keepdim=True)[0], dim = 1)
            chosen_actions = torch.multinomial(prob, 1)

            temp_R = torch.zeros(n_bandit, k, device = q_true.device)

            bandit_idx = torch.arange(n_bandit, device = device)
            chosen_idx = chosen_actions.squeeze().to(device)

            temp_R[bandit_idx, chosen_idx] = torch.randn(n_bandit, device = q_true.device) * 1 + \
            q_true[bandit_idx, chosen_idx]
            
            if n_pull > 1:
                H[bandit_idx, chosen_idx] += alpha * (temp_R[bandit_idx, chosen_idx] - Q) * \
                    (torch.ones_like(prob)[bandit_idx, chosen_idx] - prob[bandit_idx, chosen_idx])
                mask = torch.ones_like(H, dtype = torch.bool)
                mask[bandit_idx, chosen_idx] = False
                
                diff = temp_R[bandit_idx, chosen_idx] - Q
                diff_expanded = diff.unsqueeze(1).expand(-1, k)
                H[mask] -=  alpha * diff_expanded[mask] * prob[mask]
            
            Q += (temp_R[bandit_idx, chosen_idx] - Q) / n_pull
            
            avg_alpha_res += (torch.mean(temp_R[bandit_idx, chosen_idx]).item() - avg_alpha_res) / n_pull
        
        g_res_total.append(avg_alpha_res)
    
    
    g_res_total = torch.tensor(g_res_total)
    ALPHA = torch.tensor(ALPHA)
    torch.save(g_res_total, f'{current_dir}/torch_results/gradient-weighted.pth')
    torch.save(ALPHA, f'{current_dir}/torch_results/gradient-x.pth')

def greedy_optimistic_initial():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_bandit = 2000     # number of bandit
    k = 10              # number of arms
    n_pulls = 1000      # number of steps

    q_true = torch.normal(0, 1, (n_bandit, k), device = device)
    
    greedy_res_total = []       # save total results

    Q0 = [1/4, 1/2, 1, 2, 4]
    alpha = 0.1

    for val in Q0:
        Q = torch.ones((n_bandit, k), device = q_true.device) * val

        # average reward per greedy
        avg_greedy_res = 0

        for pull in range(1, n_pulls + 1):
            chosen_actions = torch.argmax(Q, dim = 1, keepdim = True)
            temp_R = torch.zeros(n_bandit, k, device = q_true.device)

            bandit_idx = torch.arange(n_bandit, device = device)
            chosen_idx = chosen_actions.squeeze().to(device)
            
            temp_R[bandit_idx, chosen_idx] = torch.randn(n_bandit, device = device) * 1 + \
            q_true[bandit_idx, chosen_idx]
            Q[bandit_idx, chosen_idx] += (temp_R[bandit_idx, chosen_idx] - Q[bandit_idx, chosen_idx]) * alpha

            avg_greedy_res += (torch.mean(temp_R[bandit_idx, chosen_idx]).item() - avg_greedy_res) / pull
        
        greedy_res_total.append(avg_greedy_res)
    
    greedy_res_total = torch.tensor(greedy_res_total)
    Q0 = torch.tensor(Q0)
    torch.save(greedy_res_total, f'{current_dir}/torch_results/greedy-weighted.pth')
    torch.save(Q0, f'{current_dir}/torch_results/greedy-x.pth')




def visualized():
    x_ticks = [1/128, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']

    num2label = {
        1/128: 1/128,
        1/64: 1,
        1/32: 2,
        1/16: 3,
        1/8: 4,
        1/4: 5,
        1/2: 6,
        1.0: 7,
        2.0: 8,
        3.0: 8.5,
        4.0: 9
    }

    fig, ax = plt.subplots()

    col = ['r-', 'k-', 'g-', 'b-']
    curve_list = ['epsilon', 'greedy', 'gradient', 'ucb']

    file_path = f'{current_dir}/torch_results'

    for i, m in enumerate(curve_list):
        y = os.path.join(file_path, f'{m}-weighted.pth')
        y = torch.load(y, map_location = torch.device('cpu')).numpy()
        x = os.path.join(file_path, f'{m}-x.pth')
        x = torch.load(x, map_location = torch.device('cpu')).numpy()

        x = [num2label[k] for k in x]

        ax.plot(x, y, col[i], linewidth = 1)
    
    # check latex package
    def is_latex_installed():
        try:
            subprocess.run(['latex', '--version'], check = True, stdout = subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    if is_latex_installed():
        plt.rc('text', usetex = True)
        ax.text(2/128, 1.3, r'$\varepsilon$-greedy', fontsize=17, color='red')
        ax.text(5, 1.25, r'gradient', fontsize=17, color='green')
        ax.text(5.3, 1.21, r'bandit', fontsize=17, color='green')
        ax.text(4.2, 1.46, r'UCB', fontsize=17, color='blue')
        ax.text(9.1, 1.45, r'greedy with', fontsize=17, color='black')
        ax.text(9.3, 1.41, r'optimistic', fontsize=17, color='black')
        ax.text(9.1, 1.37, r'initialization ', fontsize=17, color='black')
        ax.text(9.4, 1.33, r'$\alpha=0.1$', fontsize = 17, color = 'black')

        ax.text(0.3, -0.1, r'$\varepsilon$', fontsize=18, color='red', ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.1, r'$\alpha$', fontsize=18, color='green', ha='center', transform=ax.transAxes)
        ax.text(0.7, -0.1, r'$c$', fontsize=18, color='blue', ha='center', transform=ax.transAxes)
        ax.text(0.9, -0.1, r'$Q_0$', fontsize=18, color='black', ha='center', transform=ax.transAxes)
    
   

    ax.set_ylim(0.9, 1.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Average reward over last 1000 steps', fontsize = 14)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(direction = 'in', which = 'both', axis = 'both')
    
    plt.savefig(f'{current_dir}/figure.png', dpi=300, bbox_inches='tight')
    plt.show()





if __name__ == '__main__':
    import time
    t1 = time.time()
    epsilon_greedy()
    greedy_optimistic_initial()
    ucb()
    gradient_bandit()
    t2 = time.time()
    print('done, cost'.format(t2 - t1))
    visualized()
    