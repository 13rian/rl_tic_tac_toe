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
        
        # policy head
        self.fc4p = nn.Linear(27, CONST.NN_ACTION_SIZE)  # approximation for the action value function Q(s, a)
        
        # value head
        self.fc4v = nn.Linear(27, 1)                     # approximation for the value function V(s)

        # define the optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # randomly initialize all the weights
        mean = 0
        std = 0.2
        nn.init.normal_(self.fc1.weight, mean=mean, std=std)
        nn.init.normal_(self.fc2.weight, mean=mean, std=std)
        nn.init.normal_(self.fc3.weight, mean=mean, std=std)
        nn.init.normal_(self.fc4p.weight, mean=mean, std=std)
        nn.init.normal_(self.fc4v.weight, mean=mean, std=std)
         
        
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
        
        # policy
        p = self.fc4p(x)
        p = F.softmax(p, dim=1)     # values between 0 and 1 
        
        # value
        v = self.fc4v(x) 
        v = torch.tanh(v)           # values between -1 and 1 
        
        return p, v


    def train_step(self, batch, target, action_index):
        """
        executes one training step of the neural network
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target:          tensor with the true q-values estimated by the Bellmann equation
        :param action_index:    the index of the action that corresponds to the passed target
        :return:                the loss
        """

        self.train()     # allow the weights to be changed
         
        # send the tensors to the used device
        data = batch.to(Globals.device)
         
        self.optimizer.zero_grad()          # reset the gradients to zero in every epoch
        prediction = self(data)             # pass the data through the network to get the prediction

        # create the label
        label = prediction.clone()
        target = target.to(Globals.device).squeeze(1)
        label[np.arange(0, batch.shape[0]), action_index.squeeze(1)] = target     # only replace the target values of the executed action
        criterion = nn.MSELoss()
         
        # define the loss
        loss = criterion(prediction, label)
        loss.backward()              # back propagation
        self.optimizer.step()        # make one optimization step
        return loss
    
    
#         out1, out2 = model(data)
#         loss1 = criterion1(out1, target1)
#         loss2 = criterion2(out2, target2)
#         loss = loss1 + loss2
#         loss.backward()


class Agent:
    def __init__(self, learning_rate, epsilon, disc, lambda_param, batch_size, exp_buffer_size):
        """
        :param learning_rate:       learning rate for the neural network
        :param epsilon:             defines the exploration rate
        :param disc:                the discount factor
        :param lambda_param:        the lambda parameter of TD(lambda)
        :param batch_size:          the experience buffer batch size to train the training network
        :param exp_buffer_size:     the size of the experience replay buffer
        """

        self.learningRate = learning_rate                            # learning rate for the stochastic gradient decent
        self.epsilon = epsilon                                       # epsilon to choose the epsilon greedy move
        self.disc = disc                                             # the discount factor for the td update
        self.lambda_param = lambda_param                             # the lambda parameter of TD(lambda)
        self.batch_size = batch_size                                 # the size of the experience replay buffer
        self.network = Network(learning_rate).to(Globals.device)     # the neural network to train

        self.board = tic_tac_toe.BitBoard()                          # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences

        # to save the experience of one episode
        self.state_list = []
        self.action_index_list = []
        self.reward_list = []
        self.not_terminal_list = []
        self.succ_state_list = []
        self.succ_player_list = []
        self.legal_moves_list = []


    def reset_game(self):
        self.board = tic_tac_toe.BitBoard()
        
        # reset the experience lists
        self.state_list = []
        self.action_index_list = []
        self.reward_list = []
        self.not_terminal_list = []
        self.succ_state_list = []
        self.succ_player_list = []
        self.legal_moves_list = []

        

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
        
        # calculate the eligibilities recursively
        state = torch.Tensor(self.state_list).reshape(-1, CONST.NN_INPUT_SIZE)
        action_index = torch.Tensor(self.action_index_list).unsqueeze(1)
        reward = torch.Tensor(self.reward_list).unsqueeze(1)
        not_terminal = torch.Tensor(self.not_terminal_list).unsqueeze(1)
        succ_state = torch.Tensor(self.succ_state_list).reshape(-1, CONST.NN_INPUT_SIZE)
        succ_player = torch.Tensor(self.succ_player_list).unsqueeze(1)
        eligibility = self.experience_buffer.calc_eligibility(self.network, reward, succ_state, succ_player, self.legal_moves_list, self.lambda_param, self.disc)
            
        # add all the experiences of the game to the experience buffer 
        self.experience_buffer.add_batch(state, action_index, reward, not_terminal, succ_state, succ_player, self.legal_moves_list, eligibility)
            

    def epsilon_greedy_move(self):
        """
        plays the epsilon greedy move and saves the experience in the experience buffer
        :return:
        """

        # get the current state
        state, _ = self.board.bit_board_representation()
        
        # choose the move to play
        if random.random() < self.epsilon:
            # exploration
            action = self.board.random_move()
        else:
            # exploitation
            action, _ = self.board.greedy_action_move(self.network)

        action_index = action
        if self.board.player == CONST.BLACK:
            action_index = action + 9
        
        # play the epsilon greedy move
        self.board.play_move(action)
        
        # add the experience in the experience lists
        self.state_list.append(state)
        self.action_index_list.append(action_index)
        self.reward_list.append(self.board.reward())
        self.not_terminal_list.append(self.board.not_terminal_int())

        succ_state, succ_player = self.board.bit_board_representation()
        self.succ_state_list.append(succ_state)
        self.succ_player_list.append(succ_player)
        self.legal_moves_list.append(self.board.legal_moves)
    

    def q_update(self):
        """
        updates the training neural network by using a random batch from the experience replay
        :return:
        """

        # exit if the experience buffer is not yet large enough
        if self.experience_buffer.size < self.batch_size:
            return
        
        # get the random batch
        states, action_indices, eligibilities = self.experience_buffer.random_batch(self.batch_size)
        states = states.to(Globals.device)
        action_indices = action_indices.to(Globals.device)
        eligibilities = eligibilities.to(Globals.device)

        # execute the training step of the network
        self.network.train_step(states, eligibilities, action_indices)   # the eligibility trace is used as td target
        

    def update_eligibilities(self, lambda_param, disc):
        """
        refreshes the eligibilities in the experience buffer
        :param lambda_param:    lambda parameter for TD(lambda)
        :param disc:            the discount factor
        :return:
        """

        self.experience_buffer.update_eligibilities(self.network, lambda_param, disc)
    
            

    def play_against_random(self, color, game_count):
        """
        lets the agent play against a random player
        :param color:       the color of the agent
        :param game_count:  the number of games that are played
        :return:            the mean score against the random player 0: lose, 0.5 draw, 1: win
        """

        score = tic_tac_toe.q_net_against_random(self.network, color, game_count)
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
        self.eligibility = torch.empty(max_size, 1)
        
        self.size = 0                       # size of the buffer
        self.ring_index = 0                 # current index of where the next sample is added

            

    def add_batch(self, states, action_indices, rewards, not_terminals, succ_states, succ_players, succ_legal_moves, eligibilities):
        """
        adds multiple experiences to the buffer
        :param states:              the states s_t
        :param action_indices:      the action index of the nn output that was chosen in state s
        :param rewards:             the observed rewards
        :param not_terminals:       0 if the game is finished, 1 if it is not finished
        :param succ_states:         the state s_t+1 after the move was played
        :param succ_players:        the player who's move it is in state s_t+1
        :param succ_legal_moves:    all legal moves in the successor position
        :param eligibilities:       the eligibility traces
        :return:
        """

        sample_count = not_terminals.shape[0]
        start_index = self.ring_index
        end_index = self.ring_index + sample_count
        
        # check if the index is not too large
        if end_index > self.max_size:
            end_index = self.max_size
            batch_end_index = end_index - start_index
            
            # add all elements until the end of the ring buffer array
            self.add_batch(states[0:batch_end_index, :],
                           action_indices[0:batch_end_index],
                           rewards[0:batch_end_index],
                           not_terminals[0:batch_end_index],
                           succ_states[0:batch_end_index, :],
                           succ_players[0:batch_end_index],
                           succ_legal_moves[0:batch_end_index],
                           eligibilities[0:batch_end_index])
            
            # add the rest of the elements at the beginning of the buffer
            self.add_batch(states[batch_end_index:, :],
                           action_indices[batch_end_index:],
                           rewards[batch_end_index:],
                           not_terminals[batch_end_index:],
                           succ_states[batch_end_index:, :],
                           succ_players[batch_end_index:],
                           succ_legal_moves[batch_end_index:],
                           eligibilities[batch_end_index:])
            return
            
        # add the elements into the ring buffer    
        self.state[start_index:end_index, :] = states
        self.action_index[start_index:end_index] = action_indices
        self.reward[start_index:end_index] = rewards
        self.not_terminal[start_index:end_index] = not_terminals
        self.succ_state[start_index:end_index, :] = succ_states
        self.succ_player[start_index:end_index] = succ_players
        self.succ_legal_move[start_index:end_index] = succ_legal_moves
        self.eligibility[start_index:end_index] = eligibilities
          
        # update indices and size
        self.ring_index += sample_count
        self.ring_index = self.ring_index % self.max_size
          
        if self.size < self.max_size:
            self.size += sample_count
     

    def update_eligibilities(self, net, lambda_param, disc):
        """
        recalculates all the eligibility traces in the experience buffer
        :param net:             the neural network
        :param lambda_param:    lambda parameter for TD(lambda)
        :param disc:            the discount factor
        :return:
        """

        terminal_indices = np.where(self.not_terminal.data.cpu().numpy() == 0)[0]
        for i in range(terminal_indices.size - 1):
            start_index = terminal_indices[i] + 1
            end_index = terminal_indices[i+1]+1
            
            rewards = self.reward[start_index:end_index]
            succ_states = self.succ_state[start_index:end_index, :]
            succ_players = self.succ_player[start_index:end_index]
            succ_legal_moves = self.succ_legal_move[start_index:end_index]
            eligibilities = self.calc_eligibility(net, rewards, succ_states, succ_players, succ_legal_moves, lambda_param, disc)
            self.eligibility[start_index:end_index] = eligibilities
            
        # last experience can continue at the beginning of the array
        if terminal_indices[terminal_indices.size-1] < self.size-1:
            first_start_index = terminal_indices[terminal_indices.size-1]+1
            fist_end_index = self.size
            second_start_index = 0
            second_end_index = terminal_indices[0]+1
            
            rewards = torch.cat((self.reward[first_start_index:fist_end_index], self.reward[second_start_index:second_end_index]), 0)
            succ_states = torch.cat((self.succ_state[first_start_index:fist_end_index], self.succ_state[second_start_index:second_end_index]), 0)
            succ_players = torch.cat((self.succ_player[first_start_index:fist_end_index], self.succ_player[second_start_index:second_end_index]), 0)
            succ_legal_moves = self.succ_legal_move[first_start_index:fist_end_index] + self.succ_legal_move[second_start_index:second_end_index]
            eligibilities = self.calc_eligibility(net, rewards, succ_states, succ_players, succ_legal_moves, lambda_param, disc)
            self.eligibility[first_start_index:fist_end_index] = eligibilities[0:fist_end_index-first_start_index]
            self.eligibility[second_start_index:second_end_index] = eligibilities[fist_end_index-first_start_index:eligibilities.shape[0]]
             

    def calc_eligibility(self, net, rewards, succ_states, succ_players, succ_legal_moves, lambda_param, disc):
        """
        calculates the eligibilities
        :param net:                 the neural network
        :param rewards:             immediate rewards of the state transition
        :param succ_states:         the next state after the greedy move was played
        :param succ_players:        the player who's move it is in the successor state s_t+1
        :param succ_legal_moves:    all legal moves in the successor state s_t+1
        :param lambda_param:        the lambda parameter for the TD(lambda)
        :param disc:                the discount factor
        :return:                    the eligibilities
        """

        # calculate the values
        succ_states = succ_states.to(Globals.device)
        q_values = net(succ_states)
        
        # calculate all the eligibilities recursively
        exp_size = rewards.shape[0]
        eligibility = np.empty(exp_size)
        eligibility[exp_size-1] = rewards[exp_size-1]  # the terminal state
        
        for i in range(exp_size-2, -1, -1):
            if succ_players[i] == CONST.WHITE_MOVE:
                legal_q_values = q_values[0, 0:9][succ_legal_moves[i]]
                q_value, _ = legal_q_values.max(0)
            else:
                legal_q_values = q_values[0, 9:18][succ_legal_moves[i]]
                q_value, _ = legal_q_values.min(0)

            eligibility[i] = rewards[i] + disc*(lambda_param*eligibility[i+1] + (1-lambda_param)*q_value)
        
        return torch.Tensor(eligibility).unsqueeze(1)



    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:      the size of the batch
        :return:                states, players, eligibilities
        """
        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.action_index[idx, :], self.eligibility[idx, :]
