# Reinforcement Learning with Tic Tac Toe


## Goal of this Project 
The idea of this project is to try out different reinforcement learning algorithms to learn the game Tic Tac Toe. Tic Tac Toe is easy to implement and has a small state space. This is why the training of the neural networks will be fast. The goal is to use successful algorithms for more complex games such as Othello or Chess.
The different algorithms are in the folder called rl. Each subfolder contains a different reinforcement learning algorithm. In order to run the algorithms  
- run MainSelfPlayTraining.py to learn an agent how to play the game  
- run MainTestPlay.py to let the agent play against a random opponent


## Implementation Details
Tic Tac Toe is theoretically solved which means that there exists an optimal strategy that will win or at least draw against any strategy of the opponent. Furthermore, the state space is small and tabular learning could be used but since I want to use these algorithms for more complex games a neural network is used for function approximation. The performance of the agent is measured against an opponent that just plays random moves. 

The following strategies were used in all algorithms:  
- The reward of the agent is 1 if white wins, -1 if black wins and 0 if the game is still running or drawn  
- Evaluation: The reward is 1 for winning, 0.5 for draw and 0 for losing.  
If two random players are playing against each other white will score about 0.65 and black 0.35. These are the baselines to check if the network has learned something.


## Algorithms based on the Value Function
Since Tic Tac Toe is a zero sum game with perfect information it is possible to only look at the afterstates to figure out the best move from a given position. With this approach only one agent needs to be trained. In a given position all possible moves are played and the value of the successor position is calculated. The move with the highest value is chosen if the agent makes a move for white and the lowest value is chosen if the agent makes a move for black. This move is called the greedy move.  
q(s,a) = argmax V(succ(s,a)) for white
q(s,a) = argmin V(succ(s,a)) for black

The value function V is approximated by a neural network which takes the board configuration as input. The board is represented by a vector of size 18. The first 9 elements are the white pieces and the second 9 elements are the black pieces. The player that is about to move is a feature that was added in the third layer of the neural network. The feature is 0 if it is white's move and 1 if it is black's move. The neural network has only a single number V(s_t) between -1 and 1 as output that tells us how good it is to be in the current position. 1 means that white is winning and -1 means that black is winning. A value of 0 means that the position is estimated as equal. 

### TD(0) Learning
- Training of a neural network with correlated data tends to be unstable. Additionally, as the agent is learning his policy changes as well. The target to learn is therefore not stationary and convergence is not guaranteed. For this reasons the following stabilization strategies were chosen:
- replay experience: after a move was made the state transition is saved in a buffer. In every training step a batch is randomly chosen from this buffer. The batch is used to train the network in one stochastic gradient decent step
- two networks were used. A target network which is used to calculate the value function and a training network on which the training steps are performed. The target network is periodically synchronized with the training network after some time steps
- The state transitions that were induced by picking a random exploratory move were not considered in the neural network update

The algorithm is given by,

![](documentation/td0_value.png)


The ideas of this approach were taken form:  
- Reinforcement Learning - An Introduction [book](http://incompleteideas.net/book/bookdraft2017nov5.pdf)    
- Online [Lectures](https://www.youtube.com/watch?v=kZ_AUmFcZtk&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=10) from David Silver: 



### TD(&lambda) Learning
This [paper](https://arxiv.org/abs/1810.09967) introduced a new reinforcement learning algorithm that uses eligibility traces combined with an experience buffer in an efficient way. Because eligibility traces are used learning is much more efficient for longer games. Since Tic Tac Toe is a very short game with a maximum of 9 moves the difference is not that big but for a game like Othello that has a maximum of 60 moves it makes a huge difference. The algorithm was adapted for the self-play context as follows:

![](documentation/dqn_lambda_value.png)   

