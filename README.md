# ReinforcementLearning-R.S.
To reproduce the experiments in Sutton's book

# Algorithms Implemented
The following algorithms are implemented, as described in [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020trimmed.pdf) by Richard S. Sutton.

## Chapter 1 Introduction  
- Tic-Tac-Toe Game with tabular method

## Chapter 2 Multi-armed Bandits 
- Epsilon-Greedy Algorithm
- Upper Confidence bound (UCB)
- Optimistic Initial Value Method
- Gradient Bandit Algorithm
- Faster version of multi-armed bandit tasks (pytorch)
- Nonstationary version of multi-armed bandit tasks (pytorch)

![Stationary](https://github.com/erxiong0/ReinforcementLearning-R.S./blob/main/Chapter2-Multi-armed-Bandits/ordinary_version/parameter-study-of-various-bandit-methods.png)
![Nonstationary](https://github.com/erxiong0/ReinforcementLearning-R.S./blob/main/Chapter2-Multi-armed-Bandits/nonStationary_bandit_testbed/figure.png)

## Chapter 4 Dynamic Programming  
- Solutions to car rental problem
- Solutions to exercise 4.7
- Value iteration on gambler's problem

![](https://github.com/erxiong0/ReinforcementLearning-R.S./blob/main/Chapter4-DynamicProgramming/GamblersProblem/gamblerGame_value.png)

# Dependencies  
- numpy (used version 1.24.3)
- matplotlib (used version 3.7.5)
- latex
- pytorch

# Implementations  
The codes for each algorithm and corresponding plots generated can be found in the respective folders.  


# References  
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020trimmed.pdf) by Richard S. Sutton.
- [MultiArmedBandit_RL](https://github.com/SahanaRamnath/MultiArmedBandit_RL/tree/master) by SahanaRamnath
