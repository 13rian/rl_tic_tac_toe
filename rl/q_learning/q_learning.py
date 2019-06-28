import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from game import tic_tac_toe
from game.globals import Globals, CONST



class Network(nn.Module):
    def __init__(self, learning_rate):
        super(Network, self).__init__()
                
        self.fc1 = nn.Linear(CONST.NN_INPUT_SIZE, 54)    # first fully connected layer
        self.fc2 = nn.Linear(54, 54)                     # second fully connected layer
        self.fc3 = nn.Linear(54, 27)                     # third fully connected layer
        self.fc4 = nn.Linear(27, CONST.NN_ACTION_SIZE)   # approximation for the action value function Q(s, a)

        # define the optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # randomly initialize all the weights
        mean = 0
        std = 0.2
        nn.init.normal_(self.fc1.weight, mean=mean, std=std)
        nn.init.normal_(self.fc2.weight, mean=mean, std=std)
        nn.init.normal_(self.fc3.weight, mean=mean, std=std)
        nn.init.normal_(self.fc4.weight, mean=mean, std=std)
         
        
    def forward(self, x):
        # fc layer 1
        x = self.fc1(x)             
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        x = F.relu(x)
        
        # fc layer 3
        x = self.fc3(x)
        x = F.relu(x)
        
        # fc layer 4
        x = self.fc4(x)
        x = torch.tanh(x)               # values between -1 and 1 
        
        return x








    def train_step(self, batch, target, action_index):
        """
        executes one training step of the neural network
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target:          tensor with the true q-values estimated by the Bellmann equation
        :param action_index:    the index of the action that corresponds to the passed target
        :return:                the loss
        """

        self.train()  # allow the weights to be changed

        # send the tensors to the used device
        data = batch.to(Globals.device)

        self.optimizer.zero_grad()  # reset the gradients to zero in every epoch
        prediction = self(data)  # pass the data through the network to get the prediction

        # create the label
        label = prediction.clone()
        target = target.squeeze(1)
        label[np.arange(0, batch.shape[0]), action_index.squeeze(1)] = target  # only replace the target values of the executed action
        criterion = nn.MSELoss()  # use the log-likelihood loss

        # define the loss
        loss = criterion(prediction, label)
        loss.backward()  # back propagation
        self.optimizer.step()  # make one optimization step
        return loss


class Agent:
    def __init__(self, learning_rate, epsilon, disc, batch_size, exp_buffer_size):
        """
        :param learning_rate:       learning rate for the neural network
        :param epsilon:             defines the exploration rate
        :param disc:                the discount factor
        :param batch_size:          the experience buffer batch size to train the training network
        :param exp_buffer_size:     the size of the experience replay buffer
        """

        self.learningRate = learning_rate                            # learning rate for the stochastic gradient decent
        self.epsilon = epsilon                                       # epsilon to choose the epsilon greedy move
        self.disc = disc                                             # the discount factor for the td update
        self.batch_size = batch_size                                 # the size of the experience replay buffer
        self.training_network = Network(learning_rate)               # used to be trained in every step
        self.target_network = copy.deepcopy(self.training_network)   # used to calculate the targets

        self.board = tic_tac_toe.BitBoard()                          # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences

        # send the networks to the corresponding devices
        self.training_network = self.training_network.to(Globals.device)
        self.target_network = self.target_network.to(Globals.device)


    def reset_game(self):
        self.board = tic_tac_toe.BitBoard()


    def game_terminal(self):
        return self.board.terminal
    

    def play_self_play_game(self):
        """
        plays a game against itself with some exploratory moves in it
        :return:
        """

        # start a fresh game 
        self.reset_game()
        
        # play the epsilon greedy move and save the state transition in the experience lists      
        while not self.game_terminal():
            self.epsilon_greedy_move()


    def epsilon_greedy_move(self):
        """
        plays the epsilon greedy move and saves the experience in the experience buffer
        :return:
        """

        # get the current state
        state, _ = self.board.bit_board_representation()
        
        # choose the move to play
        is_exploring_move = False
        if random.random() < self.epsilon:
            # exploration
            action = self.board.random_move()
            is_exploring_move = True
        else:
            # exploitation
            action, _ = self.board.greedy_action_move(self.target_network)

        action_index = action
        if self.board.player == CONST.BLACK:
            action_index = action + 9
        
        # play the epsilon greedy move
        self.board.play_move(action)
        
        # add the experience to the experience buffer if the move was not an exploration move
        if not is_exploring_move:
            reward = self.board.reward()
            not_terminal = self.board.not_terminal_int()
            succ_state, succ_player = self.board.bit_board_representation()
            succ_legal_moves = self.board.legal_moves
            self.experience_buffer.add(state, action_index, reward, not_terminal, succ_state, succ_player, succ_legal_moves)
    

    def q_update(self):
        """
        updates the training neural network by using a random batch from the experience replay
        :return:
        """

        # exit if the experience buffer is not yet large enough
        if self.experience_buffer.size < self.batch_size:
            return
        
        # get the random batch
        states, action_indices, rewards, not_terminals, succ_states, succ_players, succ_legal_moves = self.experience_buffer.random_batch(self.batch_size)
        states = states.to(Globals.device)
        action_indices = action_indices.to(Globals.device)
        rewards = rewards.to(Globals.device)
        not_terminals = not_terminals.to(Globals.device)
        succ_states = succ_states.to(Globals.device)
        succ_players = succ_players.to(Globals.device)

        # prepare the training data
        q_values = self.target_network(succ_states)
        target = torch.empty(1, self.batch_size)
        for i in range(self.batch_size):
            if not_terminals[i] == 0:
                target[0, i] = rewards[i]
                continue

            if succ_players[i] == CONST.WHITE_MOVE:
                legal_q_values = q_values[0, 0:9][succ_legal_moves[i]]
                q_value, _ = legal_q_values.max(0)
            else:
                legal_q_values = q_values[0, 9:18][succ_legal_moves[i]]
                q_value, _ = legal_q_values.min(0)

            target[0, i] = rewards[i] + self.disc*not_terminals[i]*q_value

        # execute the training step of the network
        self.training_network.train_step(states, target, action_indices)   # the eligibility trace is used as td target


    def sync_networks(self):
        self.target_network = copy.deepcopy(self.training_network)


    def play_against_random(self, color, game_count):
        """
        lets the agent play against a random player
        :param color:       the color of the agent
        :param game_count:  the number of games that are played
        :return:            the mean score against the random player 0: lose, 0.5 draw, 1: win
        """

        score = tic_tac_toe.q_net_against_random(self.training_network, color, game_count)
        return score

        

class ExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
               
        # define the experience buffer
        self.state = torch.empty(max_size, CONST.NN_INPUT_SIZE)
        self.action_index = torch.empty([max_size, 1], dtype=torch.long)
        self.reward = -1*torch.ones(max_size, 1)
        self.not_terminal = torch.empty(max_size, 1)
        self.succ_state = torch.empty(max_size, CONST.NN_INPUT_SIZE)
        self.succ_player = torch.empty(max_size, 1)
        self.succ_legal_move = max_size*[None]
        
        self.size = 0                       # size of the buffer
        self.ring_index = 0                 # current index of where the next sample is added


    def add(self, state, action_index, reward, not_terminal, succ_state, succ_player, succ_legal_moves):
        """
        adds the passed experience to the buffer
        :param state:              the state s_t
        :param action_index:      the action index of the nn output that was chosen in state s
        :param reward:             the observed rewards
        :param not_terminal:       0 if the game is finished, 1 if it is not finished
        :param succ_state:         the state s_t+1 after the move was played
        :param succ_player:        the player who's move it is in state s_t+1
        :param succ_legal_moves:    all legal moves in the successor position
        :return:
        """

        self.state[self.ring_index, :] = torch.Tensor(state)
        self.action_index[self.ring_index] = action_index
        self.reward[self.ring_index] = reward
        self.not_terminal[self.ring_index] = not_terminal
        self.succ_state[self.ring_index, :] = torch.Tensor(succ_state)
        self.succ_player[self.ring_index] = succ_player
        self.succ_legal_move[self.ring_index] = succ_legal_moves

        # update indices and size
        self.ring_index += 1
        self.ring_index = self.ring_index % self.max_size

        if self.size < self.max_size:
            self.size += 1



    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:      the size of the batch
        :return:                state, action_index, reward, not_terminal, succ_state, succ_player, succ_legal_move
        """
        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.action_index[idx], self.reward[idx], self.not_terminal[idx], self.succ_state[idx, :], self.succ_player[idx], [self.succ_legal_move[i] for i in idx]

