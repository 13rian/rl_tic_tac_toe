import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import tic_tac_toe
from game.globals import Globals, CONST
from rl.alpha_zero.mcts import MCTS
from rl.alpha_zero import mcts



class Network(nn.Module):
    def __init__(self, learning_rate):
        super(Network, self).__init__()
                
        self.fc1 = nn.Linear(CONST.NN_INPUT_SIZE, 54)       # first fully connected layer
        self.fc2 = nn.Linear(54, 54)                        # second fully connected layer
        self.fc3 = nn.Linear(54, 27)                        # third fully connected layer
        
        # policy head
        self.fc4p = nn.Linear(27, CONST.NN_POLICY_SIZE)     # approximation for the action value function Q(s, a)
        
        # value head
        self.fc4v = nn.Linear(27, 1)                        # approximation for the value function V(s)

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


    def train_step(self, batch, target_p, target_v):
        """
        executes one training step of the neural network
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target_p:        policy target
        :param target_v:        value target
        :return:                the policy loss (value is ignored)
        """

        self.train()     # allow the weights to be changed
         
        # send the tensors to the used device
        data = batch.to(Globals.device)
         
        self.optimizer.zero_grad()                  # reset the gradients to zero in every epoch
        prediction_p, prediction_v = self(data)     # pass the data through the network to get the prediction

        # create the label
        target_p = target_p.to(Globals.device)
        target_v = target_v.to(Globals.device)
        criterion_p = nn.MSELoss()  # F.cross_entropy()
        criterion_v = nn.MSELoss()
         
        # define the loss
        loss_p = criterion_p(prediction_p, target_p)
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()              # back propagation
        self.optimizer.step()        # make one optimization step
        return loss_p
    


class Agent:
    def __init__(self, learning_rate, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size):
        """
        :param learning_rate:       learning rate for the neural network
        :param mcts_sim_count:      the number of simulations for the monte-carlo tree search
        :param c_puct:              the higher this constant the more the mcts explores
        :param temp:                the temperature, controls the policy value distribution
        :param batch_size:          the experience buffer batch size to train the training network
        :param exp_buffer_size:     the size of the experience replay buffer
        """

        self.learningRate = learning_rate                            # learning rate for the stochastic gradient decent
        self.mcts_sim_count = mcts_sim_count                         # the number of simulations for the monte-carlo tree search
        self.c_puct = c_puct                                         # the higher this constant the more the mcts explores
        self.temp = temp                                             # the temperature, controls the policy value distribution
        self.batch_size = batch_size                                 # the size of the experience replay buffer
        self.old_network = Network(learning_rate)                    # the network form the previous generation
        self.new_network = copy.deepcopy(self.old_network)           # the most actual neural network

        self.board = tic_tac_toe.BitBoard()                          # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences
        
        self.mcts = None

        # to save the experience of one episode
        self.state_list = []
        self.player_list = []
        self.policy_list = []
        self.value_list = []
        
        # send the network to the configured device
        self.old_network.to(Globals.device)
        self.new_network.to(Globals.device)


    def reset_game(self):
        self.board = tic_tac_toe.BitBoard()
        self.mcts = MCTS(self.c_puct)           # reset the search tree
        
        # reset the experience lists
        self.state_list = []
        self.player_list = []
        self.policy_list = []
        self.value_list = []
    

    def play_self_play_game(self):
        """
        plays a game against itself with some exploratory moves in it
        :return:
        """

        # start a fresh game 
        self.reset_game()
        
        # play the epsilon greedy move and save the state transition in the experience lists      
        while not self.board.terminal:
            state, player = self.board.white_perspective()
            policy = self.mcts.policy_values(self.board, self.new_network, self.mcts_sim_count, self.temp)
            
            # sample from the policy to determine the move to play
            # self.board.print()
            # print("policy: ", policy)
            move = np.random.choice(len(policy), p=policy)
            self.board.play_move(move)
            
            # save the training example
            self.state_list.append(state)
            self.player_list.append(player)
            self.policy_list.append(policy)
        
        # calculate the values from the perspective of the player who's move it is
        reward = self.board.reward()
        for player in self.player_list:
            value = reward if player == CONST.WHITE_MOVE else -reward
            self.value_list.append(value)
  
        # add the training examples to the experience buffer
        state = torch.Tensor(self.state_list).reshape(-1, CONST.NN_INPUT_SIZE)
        player = torch.Tensor(self.player_list).unsqueeze(1)
        policy = torch.Tensor(self.policy_list).reshape(-1, CONST.NN_POLICY_SIZE)
        value = torch.Tensor(self.value_list).unsqueeze(1)
             
        self.experience_buffer.add_batch(state, player, policy, value)
            


    def nn_update(self):
        """
        updates the neural network by using all samples from the experience buffer
        :return:     the average loss over all training examples
        """
        
        # get all samples in a randomized order
        states, policies, values = self.experience_buffer.random_batch(self.experience_buffer.size)
        states = states.to(Globals.device)
        policies = policies.to(Globals.device)
        values = values.to(Globals.device)
        
        avg_loss = 0
        batch_count = int(self.experience_buffer.size / self.batch_size)
        for i in range(batch_count):
            start_index = i * self.batch_size
            end_index = (i+1)*self.batch_size - 1
            
            state_batch = states[start_index:end_index, :]
            policy_batch = policies[start_index:end_index, :]
            value_batch = values[start_index:end_index, :]
            
            # execute the training step of the network
            loss = self.new_network.train_step(state_batch, policy_batch, value_batch)
            avg_loss += loss / batch_count
        
            
#         # execute the last update with a smaller batch
#         if end_index < self.experience_buffer.size - 1:
#             state_batch = states[end_index:self.experience_buffer.size - 1, :]
#             policy_batch = policies[end_index:self.experience_buffer.size - 1, :]
#             value_batch = values[end_index:self.experience_buffer.size - 1, :]
#             
#             self.new_network.train_step(state_batch, policy_batch, value_batch)
        
        self.experience_buffer.clear()      # clear the experience buffer
        
        return avg_loss
    
    
    def play_against_random(self, color, game_count):
        """
        lets the agent play against a random player
        :param color:           the color of the agent
        :param game_count:      the number of games that are played
        :return:                the mean score against the random player 0: lose, 0.5 draw, 1: win
        """

        score = tic_tac_toe.az_net_against_random(self.old_network, color, self.c_puct, self.mcts_sim_count, game_count)
        return score
    
    
    def network_duel(self, min_win_rate, game_count):
        """
        lets the old network play against the new network, if the new network has a win rate higher than
        the passed min_win_rate it will become the old network and a copy of it will be used for future training
        if the win rate is not reached the new network will be trained for another round
        :param min_win_rate       the minimal win rate to replace the old with the new network
        :param game_count:        number of games the old and the new network play against each other
        :return:                  True if the new network reached the desired win_rate, False otherwise
        """
        
        win_rate = self.play_against_old_net(game_count)
        print("win rate: ", win_rate)  
        if win_rate >= min_win_rate:
            self.old_network = copy.deepcopy(self.new_network)
            self.old_network.to(Globals.device)
            return True
        
        else:
            return False
    
    
    def play_against_old_net(self, game_count):
        """
        lets the training network play against the previous generation
        :param game_count:        number of games the old and the new network play against each other
        :param c_puct:            constant that controls the exploration rate of the mcts
        :param mcts_sim_count:    number of monte-carlo tree search simulations
        :return:         
        """

        half_game_count = int(game_count/2)
        wins_old_net = 0
        wins_new_net = 0
         
        # play half the games where the old network is white
        board = tic_tac_toe.BitBoard()
        mcts_old = mcts.MCTS(self.c_puct)
        mcts_new = mcts.MCTS(self.c_puct)
        for _ in range(half_game_count):
            board = tic_tac_toe.BitBoard()
            while not board.terminal:
                if board.player == CONST.WHITE:
                    policy = mcts_old.policy_values(board, self.old_network, self.mcts_sim_count, 0)
                    move = np.where(policy==1)[0] 
                    board.play_move(move)
                else:
                    policy = mcts_new.policy_values(board, self.new_network, self.mcts_sim_count, 0)
                    move = np.where(policy==1)[0] 
                    board.play_move(move)
        
                # board.print()
            if board.reward() == 1:
                wins_old_net += 1
                
            if board.reward() == -1:
                wins_new_net += 1
                
        
        # play half the games where the new network is white
        for _ in range(half_game_count):
            board = tic_tac_toe.BitBoard()
            while not board.terminal:
                if board.player == CONST.WHITE:
                    policy = mcts_new.policy_values(board, self.new_network, self.mcts_sim_count, 0)
                    move = np.where(policy==1)[0] 
                    board.play_move(move)
                else:
                    policy = mcts_old.policy_values(board, self.old_network, self.mcts_sim_count, 0)
                    move = np.where(policy==1)[0] 
                    board.play_move(move)
            
                # board.print()
            if board.reward() == 1:
                wins_new_net += 1
                
            if board.reward() == -1:
                wins_old_net += 1
                
        if wins_new_net == 0:
            return 0
        else:
            return wins_new_net / (wins_new_net + wins_old_net)       

    
     

class ExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
               
        self.state = torch.empty(max_size, CONST.NN_INPUT_SIZE)
        self.player = torch.empty(max_size, 1)
        self.policy = torch.empty(max_size, CONST.NN_POLICY_SIZE)
        self.value = torch.empty(max_size, 1)
        
        self.size = 0                  # size of the buffer
        self.ring_index = 0            # current index of where the next sample is added
       
        
    def clear(self):
        """
        empties the experience buffer
        """
        self.state = torch.empty(self.max_size, CONST.NN_INPUT_SIZE)
        self.player = torch.empty(self.max_size, 1)
        self.policy = torch.empty(self.max_size, CONST.NN_POLICY_SIZE)
        self.value = torch.empty(self.max_size, 1)
        
        self.size = 0                  # size of the buffer
        self.ring_index = 0            # current index of where the next sample is added
        
        
        
    def add_batch(self, states, players, policies, values):
        """
        adds the multiple experiences to the buffer
        :param states:           the state s_t
        :param players:          the player who's move it is
        :param policies:         probability value for all actions
        :param values:           value of the current state
        :return:
        :return:
        """

        sample_count = values.shape[0]
        start_index = self.ring_index
        end_index = self.ring_index + sample_count
        
        # check if the index is not too large
        if end_index > self.max_size:
            end_index = self.max_size
            batch_end_index = end_index - start_index
            
            # add all elements until the end of the ring buffer array
            self.add_batch(states[0:batch_end_index, :],
                           players[0:batch_end_index],
                           policies[0:batch_end_index, :],
                           values[0:batch_end_index])
            
            # add the rest of the elements at the beginning of the buffer
            self.add_batch(states[batch_end_index:, :],
                           players[batch_end_index:],
                           policies[batch_end_index:, :],
                           values[batch_end_index:])
            return
            
        # add the elements into the ring buffer    
        self.state[start_index:end_index, :] = states
        self.player[start_index:end_index] = players
        self.policy[start_index:end_index, :] = policies
        self.value[start_index:end_index] = values
          
        # update indices and size
        self.ring_index += sample_count
        self.ring_index = self.ring_index % self.max_size
          
        if self.size < self.max_size:
            self.size += sample_count
        
            

    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:   the size of the batch
        :return:             state, policy, value
        """

        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.policy[idx, :], self.value[idx]