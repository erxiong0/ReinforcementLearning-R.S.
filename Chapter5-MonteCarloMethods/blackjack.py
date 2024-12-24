#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# 2024 Chichi Syun(chichi007@163.com)                                 #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Tuple
import os


# actions: hit or stick
HIT, STICK = 0, 1
ACTIONS = [HIT, STICK]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype = int)
for i in range(12, 20):
    POLICY_PLAYER[i] = HIT
POLICY_PLAYER[20] = STICK
POLICY_PLAYER[21] = STICK

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = HIT
for i in range(17, 22):
    POLICY_DEALER[i] = STICK


# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[ player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    if np.random.binomial(1, 0.5) == 1:
        return STICK
    return HIT


# get a new card
def get_card() -> int:
    card = np.random.randint(1, 14)
    return min(card, 10)

# get the value of a card (11 for ace)
def card_value(card_id: int) -> int:
    return 11 if card_id == 1 else card_id


# policy_player: specify policy for player
# initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# initial_action: the initial action
def judger(policy_player: Callable, initial_state: List = None, initial_action: int = None):
    # player
    player_sum = 0
    player_trajectory = []      # store cards
    usable_ace_player = False   # whether use Ace as 11

    # dealer
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        while player_sum < 12:
            card = get_card()
            player_sum += card_value(card)
            
            # if the player's sum is larger than 21, he may hold 1 or 2 aces.
            if player_sum > 21:
                player_sum -= 10
            else:
                usable_ace_player |= (1 == card)
        
        dealer_card1 = get_card()
        dealer_card2 = get_card()
    
    else:
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()
    
    # initial state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    if dealer_sum > 21:
        dealer_sum -= 10
    
    assert dealer_sum <= 21
    assert player_sum <= 21

    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            action = policy_player(usable_ace_player, player_sum, dealer_card1)
        
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        
        if action == STICK:
            break

        card = get_card()
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        
        player_sum += card_value(card)

        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1

        if player_sum > 21:
            return state, -1, player_trajectory
        
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)
    

    # dealer's turn
    while True:
        action = POLICY_DEALER[dealer_sum]
        if action == STICK:
            break

        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)

        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        
        if dealer_sum > 21:
            return state, 1, player_trajectory
        
        usable_ace_dealer = (ace_count == 1)
    
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

 
# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes: int):
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for _ in tqdm(range(0, episodes)):
        _, reward, player_trajectory = judger(target_policy_player)
        for state, _ in player_trajectory:
            usable_ace, player_sum, dealer_card = state
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    
    return states_usable_ace/states_no_usable_ace_count, states_no_usable_ace/states_no_usable_ace_count


# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes: int):
    state_action_values = np.zeros((10, 10, 2, 2))      # (playerSum, dealerCard, usableAce, action)
    state_action_pair_count = np.ones((10, 10, 2, 2))   # initialze counts to 1 to avoid division by 0

    # greedy
    def behavior_policy(usable_ace: bool, player_sum: int, dealer_card: int):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns (s, a)
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                state_action_pair_count[player_sum, dealer_card, usable_ace, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    # epsilon
    for episode in tqdm(range(episodes)):
        initial_state = [bool(np.random.choice([0, 1])), 
                         np.random.choice(range(12, 22)), 
                         np.random.choice(range(1, 11))]
        
        initial_action = np.random.choice(ACTIONS)
        current_policy = behavior_policy if episode else target_policy_player
        _, reward, trajectory = judger(current_policy, initial_state, initial_action)
        first_visit_check = set()
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            
            # update values of state-action pairs
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1
    
    return state_action_values / state_action_pair_count


# Monte Carlo Sample with off-policy
def monte_carlo_off_policy(episodes: int):
    intial_state = [True, 13, 2]

    rhos = []
    returns = []

    for _ in range(episodes):
        _, reward, player_trajectory = judger(behavior_policy_player, initial_state = intial_state)

        numerator = 1.0
        denominator = 1.0
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
        
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)
    
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns

    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)
    
    return ordinary_sampling, weighted_sampling


def plot_value_function_3d(states: List, titles: List[str], figsize: Tuple[int] = (4, 2), name: str = '3d'):
    """
    Plots the value function as a surface plot on a single figure with four subplots.
    """
    fig = plt.figure(figsize=figsize)

    for i, (state, title) in enumerate(zip(states, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        # Assuming state is a 2D numpy array
        X, Y = np.meshgrid(range(1, 11), range(12, 22))
        Z = state

        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Dealer Showing', fontsize=10)
        ax.set_ylabel('Player Sum', fontsize=10)
        ax.set_xlim(0,10)
        ax.set_zlabel('Value', fontsize=10)
        ax.set_title(title, fontsize=12)
        # ax.view_init(elev=30, azim=-120)
        ax.view_init(elev=20, azim=-45)

    fig.colorbar(surf, ax=fig.axes, shrink=0.75, aspect=30)

    current_dir = os.getcwd()
    plt.savefig(f'{current_dir}/{name}.png')
    plt.show()

def figure_on_policy_3d():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
               'Usable Ace, 500000 Episodes',
               'No Usable Ace, 10000 Episodes',
               'No Usable Ace, 500000 Episodes']

    plot_value_function_3d(states, titles, figsize=(20, 10), name = "monte_carlo_on_policy")


def figure_monte_carlo_es_3d():
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis = -1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis = -1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis = -1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis = -1)
    
    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]
    
    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']
    
    plot_value_function_3d(images, titles, figsize=(20, 10), name = 'monte_carlo_es')


def figure_monte_carlo_off_policy():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)

    for _ in tqdm(range(runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs
    
    fig, ax = plt.subplots(figsize = (6,4))
    ax.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    ax.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    ax.set_ylim(-0.1, 5)
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel(f'Mean square error\n(average over {runs} runs)')
    ax.set_xscale('log')
    ax.legend(loc ='best')

    current_dir = os.getcwd()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'{current_dir}/mean_square_error.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    figure_on_policy_3d()
    figure_monte_carlo_es_3d()
    figure_monte_carlo_off_policy()