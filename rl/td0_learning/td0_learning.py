import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import tic_tac_toe
from game.globals import Globals, CONST
   

class Network(nn.Module):
    def __init__(self, learning_rate):
        super(Network, self).__init__()
                
        self.fc1 = nn.Linear(CONST.NN_INPUT_SIZE, 54)    # first fully connected layer
        self.fc2 = nn.Linear(54, 27)                     # second fully connected layer
        self.fc3 = nn.Linear(28, 9)                      # add what player is about to move to this feature vector
        self.fc4 = nn.Linear(9, 1)                       # approximation for the value function V(s)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # randomly initialize all the weights
        mean = 0
        std = 0.2
        nn.init.normal_(self.fc1.weight, mean=mean, std=std)
        nn.init.normal_(self.fc2.weight, mean=mean, std=std)
        nn.init.normal_(self.fc3.weight, mean=mean, std=std)
        nn.init.normal_(self.fc4.weight, mean=mean, std=std)
         
        
    def forward(self, x, players):
        # fc layer 1
        x = self.fc1(x)             
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        x = F.relu(x)
        
        # fc layer 3
        x = torch.cat((x, players), dim=1)
        x = self.fc3(x)
        x = F.relu(x)
        
        # fc layer 4
        x = self.fc4(x)
        x = torch.tanh(x)               # values between -1 and 1 
        
        return x


    def train_step(self, batch, player, target):
        """
        executes one training step of the neural network
        :param batch:   tensor with data [batchSize, nn_input_size]
        :param player:  the player who's move it is
        :param target:  tensor with the true v-value estimated by the Bellmann equation
        :return:        the loss
        """

        self.train()     # allow the weights to be changed
         
        # send the tensors to the used device
        data = batch.to(Globals.device)
        player = player.to(Globals.device)
        label = target.to(Globals.device)
         
        self.optimizer.zero_grad()          # reset the gradients to zero in every epoch
        prediction = self(data, player)     # pass the data through the network
        criterion = nn.MSELoss()            # use the log-likelihood loss
         
        # define the loss
        loss = criterion(prediction, label)
        loss.backward()                     # back propagation
        self.optimizer.step()               # make one optimization step
        return loss


class Agent:
    def __init__(self, learning_rate, epsilon, disc, batch_size, exp_buffer_size):
        """
        :param learning_rate:    learning rate for the neural network
        :param epsilon:          defines the exploration rate
        :param disc:             the discount factor
        :param batch_size:       the batch size to train the training network
        :param exp_buffer_size:  the size of the experience replay buffer
        """

        self.learningRate = learning_rate                             # learning rate for the stochastic gradient decent
        self.epsilon = epsilon                                        # epsilon to choose the epsilon greedy move
        self.disc = disc                                              # the discount factor for the q update
        self.batch_size = batch_size                                  # the size of the experience replay buffer
        self.training_network = Network(learning_rate)                # used to be trained in every step
        self.target_network = copy.deepcopy(self.training_network)    # used to calculate the targets
        
        self.board = tic_tac_toe.BitBoard()                           # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)    # buffer that saves all experiences
        
        # send the networks to the corresponding devices
        self.training_network = self.training_network.to(Globals.device)
        self.target_network = self.target_network.to(Globals.device)
        

    def reset_board(self):
        self.board = tic_tac_toe.BitBoard()
        

    def game_terminal(self):
        return self.board.terminal
            

    def epsilon_greedy_move(self):
        """
        plays the epsilon greedy move and saves the experience in the experience buffer
        :return:
        """

        # get the current state
        state, player = self.board.bit_board_representation()
        
        # choose the move to play
        is_exploring_move = False
        if random.random() < self.epsilon:
            # exploration
            action = self.board.random_move()
            is_exploring_move = True
        else:
            # exploitation
            action, _ = self.board.greedy_value_move(self.target_network)
        
        # play the epsilon greedy move
        self.board.play_move(action)
        
        # add the experience to the experience buffer if the move was not an exploration move
        if not is_exploring_move:
            reward = self.board.reward()
            not_terminal = self.board.not_terminal_int()
            succ_state, succ_player = self.board.bit_board_representation()
            self.experience_buffer.add(state, player, reward, not_terminal, succ_state, succ_player)
    

    def td_update(self):
        """
        updates the training neural network by using a random batch from the experience replay
        :return:
        """

        # exit if the experience buffer is not yet large enough
        if self.experience_buffer.size < self.batch_size:
            return
        
        # get the random batch
        states, players, rewards, not_terminals, succ_states, succ_players = self.experience_buffer.random_batch(self.batch_size)
        states = states.to(Globals.device)
        players = players.to(Globals.device)
        rewards = rewards.to(Globals.device)
        not_terminals = not_terminals.to(Globals.device)
        succ_states = succ_states.to(Globals.device)
        succ_players = succ_players.to(Globals.device)
        
        # prepare the training data
        values = self.target_network(succ_states, succ_players)
        td_target = rewards + self.disc*not_terminals*values

        # execute the training step of the network
        self.training_network.train_step(states, players, td_target)
        

    def sync_networks(self):
        self.target_network = copy.deepcopy(self.training_network)
            

    def play_against_random(self, color, game_count):
        """
        lets the agent play against a random player
        :param color:       the color of the agent
        :param game_count:  the number of games that are played
        :return:            the mean score against the random player 0: lose, 0.5 draw, 1: win
        """

        score = tic_tac_toe.v_net_against_random(self.training_network, color, game_count)
        return score


class ExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
               
        # define the experience buffer
        self.state = torch.empty(max_size, CONST.NN_INPUT_SIZE)
        self.player = torch.empty(max_size, 1)
        self.reward = torch.empty(max_size, 1)
        self.not_terminal = torch.empty(max_size, 1)
        self.succ_state = torch.empty(max_size, CONST.NN_INPUT_SIZE)
        self.succ_player = torch.empty(max_size, 1)
        
        self.size = 0                  # size of the buffer
        self.ring_index = 0            # current index of where the next sample is added
        

    def add(self, state, player, reward, not_terminal, succ_state, succ_player):
        """
        adds the passed experience to the buffer
        :param state:           the state s_t
        :param player:          the player who'S move it is
        :param reward:          the observed reward
        :param not_terminal:    0 if the game is finished, 1 if it is not finished
        :param succ_state:      the state after the action was executed, s_t+1
        :param succ_player:     the player who'S move it is in the successor state s_t+1
        :return:
        """

        self.state[self.ring_index, :] = torch.Tensor(state)
        self.player[self.ring_index] = player
        self.reward[self.ring_index] = reward
        self.not_terminal[self.ring_index] = not_terminal
        self.succ_state[self.ring_index, :] = torch.Tensor(succ_state)
        self.succ_player[self.ring_index] = succ_player
        
        # update indices and size
        self.ring_index += 1
        self.ring_index = self.ring_index % self.max_size
        
        if self.size < self.max_size:
            self.size += 1
            

    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:   the size of the batch
        :return:             state, player, reward, not_terminal, succ_state, succ_player
        """

        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.player[idx], self.reward[idx], self.not_terminal[idx], self.succ_state[idx, :], self.succ_player[idx]
