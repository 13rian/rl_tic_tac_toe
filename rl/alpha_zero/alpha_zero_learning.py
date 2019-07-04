import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from game import tic_tac_toe
from game.globals import Globals, CONST
from rl.alpha_zero.mcts import MCTS



class Network(nn.Module):
    def __init__(self, learning_rate):
        super(Network, self).__init__()
                
        self.fc1 = nn.Linear(CONST.NN_INPUT_SIZE + 1 , 54)  # first fully connected layer, all stones + the color of the player
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
        self.new_network = copy.deepcopy(self.training_network)      # the most actual neural network

        self.board = tic_tac_toe.BitBoard()                          # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences
        
        self.mcts = None

        # to save the experience of one episode
        self.state_list = []
        self.policy_list = []
        self.value_list = []
        
        # send the network to the configured device
        self.old_network.to(Globals.device)
        self.new_network.to(Globals.device)


    def reset_game(self):
        self.board = tic_tac_toe.BitBoard()
        self.mcts = MCTS(self.board, self.c_puct)
        
        # reset the experience lists
        self.state_list = []
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
            state = self.board.to_feature_vector()
            policy = self.mcts.policy_values(self.new_network, self.mcts_sim_count, self.temp)
            
            # sample from the policy to determine the move to play
            move = np.random.choice(len(policy), p=policy)
            self.board.play_move(move)
            
            # save the training example
            self.state_list.append(state)
            self.policy_list.append(policy)
        
        # calculate the values from the perspective of the player who's move it is
        reward = self.board.reward()
        for state in self.state_list:
            player = state[CONST.NN_INPUT_SIZE]
            value = reward if player == CONST.WHITE_MOVE else -reward
            self.value_list.append(value)
  
        # add the training examples to the experience buffer
        state = torch.Tensor(self.state_list).reshape(-1, CONST.NN_INPUT_SIZE + 1)
        policy = torch.Tensor(self.action_index_list).reshape(-1, CONST.NN_POLICY_SIZE)
        value = torch.Tensor(self.reward_list).unsqueeze(1)
             
        self.experience_buffer.add_batch(state, policy, value)
            


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
               
        self.state = torch.empty(max_size, CONST.NN_INPUT_SIZE + 1)
        self.policy = torch.empty(max_size, CONST.NN_POLICY_SIZE)
        self.value = torch.empty(max_size, 1)
        
        self.size = 0                  # size of the buffer
        self.ring_index = 0            # current index of where the next sample is added
        
        
    def add_batch(self, states, policies, values):
        """
        adds the multiple experiences to the buffer
        :param states:           the state s_t
        :param policies:          probability value for all actions
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
                           policies[0:batch_end_index, :],
                           values[0:batch_end_index])
            
            # add the rest of the elements at the beginning of the buffer
            self.add_batch(states[batch_end_index:, :],
                           policies[batch_end_index:, :],
                           values[batch_end_index:])
            return
            
        # add the elements into the ring buffer    
        self.state[start_index:end_index, :] = states
        self.policy[start_index:end_index, :] = policies
        self.value[start_index:end_index] = values
          
        # update indices and size
        self.ring_index += sample_count
        self.ring_index = self.ring_index % self.max_size
          
        if self.size < self.max_size:
            self.size += sample_count
        
        
        
#         
# 
#     def add(self, state, policy, value):
#         """
#         adds the passed experience to the buffer
#         :param state:           the state s_t
#         :param policy:          probability value for all actions
#         :param value:           value of the current state
#         :return:
#         """
# 
#         self.state[self.ring_index, :] = torch.Tensor(state)
#         self.policy[self.ring_index, :] = torch.Tensor(policy)
#         self.value[self.ring_index] = value
#         
#         
#         # update indices and size
#         self.ring_index += 1
#         self.ring_index = self.ring_index % self.max_size
#         
#         if self.size < self.max_size:
#             self.size += 1
            

    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:   the size of the batch
        :return:             state, policy, value
        """

        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.policy[idx, :], self.value[idx]