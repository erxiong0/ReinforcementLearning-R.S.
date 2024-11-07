import gym
import numpy as np
import pickle   # save model
from typing import Tuple, Dict, Any
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# environment
class TicTacToe(gym.Env):
    def __init__(self):
        self.state = np.array([0] * 9)  # initial state space
        s1 = self._get_all_states(1)
        s2 = self._get_all_states(-1)
        s1.update(s2)
        self.policy_space = s1          # initial policy space
    

    def reset(self):
        self.state = np.array([0] * 9)
        return self.state
    
    def step(self, action: int, current_symbol: int) -> Tuple[np.array, int, bool, int]:
        """Update state"""
        self.state[action] = current_symbol
        done, winner = self._check_outcome(self.state)
        
        if winner == current_symbol:
            reward = 1
        elif winner == 0:
            reward = 0.5
        else:
            reward = 0
        
        return self.state, reward, done, winner

    
    def _check_outcome(self, state: np.array) -> Tuple[bool, int]:
        """Check wheter game is over
        :param state: current state, e.g. `[1,1,1,0,-1,-1,-1,0,0]`
        
        :return a tuple to tell whether the game is over and if it is over, who is the winner. 
            e.g. `[1,1,1,0,-1,-1,-1,0,0] --> (True, 1)`, game is over, player 1 is winner
                `[1,-1,-1,1,-1,1,-1,1,-1] --> (True, 0)`, game is over, no one wins
                `[0,0,0,0,0,0,0,0,1] --> (False, 0)` game is still going, no one wins
        """
        done = False

        # check diagonal
        diag1 = sum([state[0], state[4], state[8]])
        diag2 = sum([state[2], state[4], state[6]])
        if diag1 == 3 or diag2 == -3 or diag2 == 3 or diag2 == -3:
            done = True
            return done, state[4]

        # check row
        for i in range(0, 9, 3):
            row_sum = sum(state[i: i+3])
            if row_sum == 3 or row_sum == -3:
                done = True
                return done, state[i]
        
        # check column
        for i in range(0, 3):
            col_sum = sum([state[i], state[i + 3],  state[i + 6]])
            if col_sum == 3 or col_sum == -3:
                done = True
                return done, state[i]
        
        # no one wins
        return done if 0 in state else True, 0

    
    def _get_all_states(self, current_symbol: int) -> Dict[Tuple[bool, int], Any]:
        """Numerate all possible states of first palyer `current_symbol` and store in a dictionary
        :param current_symbol: `int` represent player, either `1` or `-1`

        :return dictionary, key is a state (in string format) and value is outcome for such state. e.g
            [1, 0, 0, 0, 0, 0, 0, 0, 0] --> {`100000000`: (False, 0)}
        """
        current_state = np.array([0] * 9)
        all_states = dict()
        def _get_all_states_impl(current_state, current_symbol, all_states):
            for i in range(9):
                if current_state[i] == 0:
                    next_state = np.copy(current_state)
                    next_state[i] = current_symbol
                    # transfer state into string format
                    next_state_hash = "".join(map(lambda x: str(x), next_state))  
                    # if state not saved before
                    if next_state_hash not in all_states.keys():
                        # check outcome 
                        done, winner = self._check_outcome(next_state)
                        # save information
                        all_states[next_state_hash] = (done, winner)
                        if not done:
                            _get_all_states_impl(next_state, -current_symbol, all_states)
        
        _get_all_states_impl(current_state, current_symbol, all_states)
        return all_states
        


class Agent:
    def __init__(self,
                all_states: Dict = {},
                policy: str = 'epsilon-greedy', 
                symbol: int = 1, 
                epsilon: float = 0.1, 
                step_size: float = 0.01,
                save_model_path: str = ".",
                load_model_path: str = ""):
        
        self.estimations = {}       # record eatimate value of each state,
                                    # the value will update after each greedy move
        self.states = []            
        self.greedy = []            # save greedy result
        self.step_size = step_size  # learning config
        self.epsilon = epsilon      # epsilon-greedy method
        self.symbol = symbol        # to distinguish player
        self.policy = policy        # learning strategy
        if load_model_path == "":
            self._initial_estimations(all_states)
        else:
            self.load_policy(load_model_path)

        self.path = save_model_path
    
    def _initial_estimations(self, all_states: dict):   # initial estimates of all possible states
        for k, v in all_states.items():
            (done, winner) = v
            if done:
                if winner == self.symbol:
                    self.estimations[k] = 1.0
                elif winner == 0:
                    self.estimations[k] = 0.5
                else:
                    self.estimations[k] = 0
            else:
                self.estimations[k] = 0.5
    
    def reset(self):
        self.states = []
        self.greedy = []
    
    def choose_action(self, state):
        # imperfect opponent
        if self.policy == 'random':
            return self._explore_search(state)
        
        # greedy
        elif self.policy == 'epsilon-greedy':
            return self._epsilon_greedy_search(state)

    def _explore_search(self, state):
        next_pos = np.where(state == 0)[0]
        if len(next_pos) > 1:
            action = random.choice(next_pos)
        else:
            action = next_pos[0]
        return action
    
    def _epsilon_greedy_search(self, state):
        self.states.append(state)
        next_states = []
        
        # enumerate all possible next action
        next_pos = np.where(state == 0)[0].tolist()
        for i in next_pos:
            t_state = np.copy(state)
            t_state[i] = self.symbol
            next_states.append(self._hash_state(t_state))
        
        # epsilon-greedy method
        if random.random() < self.epsilon:
            action = random.choice(next_pos)
            self.greedy.append(False)
            return action

        values = []
        for s, idx in zip(next_states, next_pos):
            values.append((self.estimations[s], idx))
        
        np.random.shuffle(values)
        values.sort(key = lambda x: x[0], reverse = True)
        action = values[0][1]
        self.greedy.append(True)
        return action
    
    def _hash_state(self, state):
        return "".join(map(lambda x: str(x), state))

    def backup(self):
        for i in reversed(range(len(self.states) - 1)):
            state = self._hash_state(self.states[i])
            next_state = self._hash_state(self.states[i + 1])
            td_error = self.greedy[i] * (self.estimations[next_state] - self.estimations[state])
            # update estimation with weighted-sample method
            self.estimations[state] += self.step_size * td_error

    # save result
    def save_policy(self, path: str = ""):
        if path == "":
            path = self.path
        with open(f'{path}/policy-{self.symbol}.bin', 'wb') as f:
            pickle.dump(self.estimations, f)
        
    # load policy
    def load_policy(self, model_path):
        with open(model_path, 'rb') as f:
            self.estimations = pickle.load(f)


class Judger:
    def __init__(self, player1: Agent, player2: Agent, env: TicTacToe, random_first_hand: bool = False):
        self.p1 = player1
        self.p2 = player2
        self.env = env
        self.current_state = env.state
        self.random_first_hand = random_first_hand
        if self.random_first_hand:      # random choose who plays first
            self._first_hand()
    
    def reset(self):
        self.env.reset()
        self.p1.reset()
        self.p2.reset()
        if self.random_first_hand:
            self._first_hand()
        
    def _first_hand(self):
        s = [self.p1, self.p2]
        np.random.shuffle(s)
        self.p1 = s[0]
        self.p2 = s[1]
    
    
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self):
        alternator = self.alternate()
        self.reset()
        current_state, reward = self.env.state, 0
        done = False
        winner = 0
        state_history = []
        state_history.append(np.copy(current_state))
        while not done:
            player = next(alternator)
            action = player.choose_action(current_state)
            next_state, reward, done, winner = self.env.step(action, player.symbol)
            current_state = next_state
            state_history.append(np.copy(current_state))
        
        return winner, state_history


def train(judger: Judger, epochs: int = 10000):
    player1_win = 0.0
    player2_win = 0.0

    for i in range(1, epochs + 1):
        winner, _ = judger.play()
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        
        print(f'Epoch {i}, player 1 win {player1_win / i}, player 2 win {player2_win / i}')
        player1.backup()
    
    player1.save_policy()


def compete(judger: Judger, turns: int, visulization: bool = False, save_image_path: str = "."):
    player1_win = 0.0
    player2_win = 0.0

    for _ in range(1, 1 + turns):
        winner, state_history = judger.play()
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        judger.reset()
    
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win, player2_win))
    
    # visualize last state_history
    if visulization:
        visualized_game(state_history, save_image_path)
    


def visualized_game(history: list, save_image_path: str):
    fig = plt.figure(figsize = (5, 5))
    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    plt.plot([0, 3], [1, 1], color = 'black', linewidth = 2)
    plt.plot([0, 3], [2, 2], color = 'black', linewidth = 2)
    plt.plot([1, 1], [0, 3], color = 'black', linewidth = 2)
    plt.plot([2, 2], [0, 3], color = 'black', linewidth = 2)
    plt.tick_params(axis = 'both', which = 'both',\
                    bottom = False, top = False, right = False, left = False,\
                    labelbottom = False, labelleft = False)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    scat1 = ax.scatter([], [], marker = '*', color = 'orange', s = 60 ** 2)
    scat2 = ax.scatter([], [], marker = 'o', color = 'purple', s = 60 ** 2)

    def init():
        scat1.set_offsets(np.empty((0, 2)))
        scat2.set_offsets(np.empty((0, 2)))
        return (scat1, scat2)

    def animate(i):
        x1, y1, x2, y2 = [], [], [], []
        state_history_solo = history[i]
        for j in range(9):
            if state_history_solo[j] == 1:
                x1.append(j % 3 + 0.5)
                y1.append(2.5 - int(j / 3))
            elif state_history_solo[j] == -1:
                x2.append(j % 3 + 0.5)
                y2.append(2.5 - int(j / 3))
        if x1 and y1:
            scat1.set_offsets(list(zip(x1, y1)))
        if x2 and y2:
            scat2.set_offsets(list(zip(x2, y2)))
        
        return (scat1, scat2)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = len(history), interval = 200, repeat = False)
    anim.save(f'{save_image_path}/tic_tac_toe.gif', writer='imagemagick')


if __name__ == '__main__':
    """train"""
    env = TicTacToe()                       # initial environment
    all_states = env.policy_space           # get all possible states

    # palyer 1
    player1 = Agent(all_states = all_states, 
                    policy = 'epsilon-greedy', 
                    symbol = 1,
                    epsilon = 0.1,
                    step_size = 0.01,
                    save_model_path='./Chapter1')
    
    # player 2: imperfect opponent
    player2 = Agent(policy = 'random', 
                    symbol = -1)    

    judger = Judger(player1, player2, env, True)    # initial judger
    epochs = 10000                                  # learning epochs

    train(judger, epochs)

    """compete"""
    # env = TicTacToe()
    # model_path = "./Chapter1/policy-1.bin"
    # player1 = Agent(load_model_path = model_path)
    # player2 = Agent(policy = 'random', symbol= -1)
    # judger = Judger(player1, player2, env)
    # turns = 10
    # image_path = '.'
    # compete(judger, turns, True, image_path)


