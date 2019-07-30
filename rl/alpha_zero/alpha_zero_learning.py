import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import tic_tac_toe
from game.globals import Globals, CONST
from game import minimax
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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        
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
        :return:                policy loss, value loss
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
        return loss_p, loss_v
    


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


    def clear_exp_buffer(self):
        self.experience_buffer.clear()


    def reset_game(self):
        self.board = tic_tac_toe.BitBoard()
        self.mcts = MCTS(self.c_puct)           # reset the search tree
        
        # reset the experience lists
        self.state_list = []
        self.player_list = []
        self.policy_list = []
        self.value_list = []
    

    def play_self_play_game(self, temp_threshold):
        """
        :param temp_threshold:  up to this move the temp will be temp, after the threshold it will be set to 0
        plays a game against itself with some exploratory moves in it
        :return:
        """

        # start a fresh game 
        self.reset_game()
        
        # play the epsilon greedy move and save the state transition in the experience lists
        move_count = 0      
        while not self.board.terminal:
            state, player = self.board.white_perspective()
            temp = 0 if move_count >= temp_threshold else self.temp
            policy = self.mcts.policy_values(self.board, self.new_network, self.mcts_sim_count, temp)
            
            # sample from the policy to determine the move to play
            # self.board.print()
            # print("policy: ", policy)
            move = np.random.choice(len(policy), p=policy)
            self.board.play_move(move)
            
            # save the training example
            self.state_list.append(state)
            self.player_list.append(player)
            self.policy_list.append(policy)
            move_count += 1
        
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
            


    def nn_update(self, update_count):
        """
        updates the neural network by picking a random batch form the experience replay
        :param update_count:    defines how many time the network is updated
        :return:                average policy and value loss over all mini batches
        """

        avg_loss_p = 0
        avg_loss_v = 0
        for i in range(update_count):
            # get a random batch
            states, policies, values = self.experience_buffer.random_batch(self.batch_size)
            states = states.to(Globals.device)
            policies = policies.to(Globals.device)
            values = values.to(Globals.device)

            # train the net with one mini-batch
            loss_p, loss_v = self.new_network.train_step(states, policies, values)
            avg_loss_p += loss_p
            avg_loss_v += loss_v

        # calculate the mean of the loss
        avg_loss_p /= update_count
        avg_loss_v /= update_count

        return avg_loss_p.item(), avg_loss_v.item()

    
    
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
        :return:                  the win rate
        """

        score_old, score_new = net_vs_net(self.old_network, self.new_network, game_count, self.mcts_sim_count, self.c_puct, 0)
        win_rate = score_new / (score_old + score_new)
        return win_rate

     

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


def net_vs_net(net1, net2, game_count, mcts_sim_count, c_puct, temp):
    """
    lets two alpha zero networks play against each other
    :param net1:            net for player 1
    :param net2:            net for player 2
    :param game_count:      total games to play
    :param mcts_sim_count   number of monte carlo simulations
    :param c_puct           constant that controls the exploration
    :param temp             the temperature
    :return:                score of network1, score of network2
    """

    half_game_count = int(game_count / 2)
    net1_score = 0
    net2_score = 0

    mcts1 = mcts.MCTS(c_puct)
    mcts2 = mcts.MCTS(c_puct)
    for _ in range(half_game_count):
        # play half the games where net1 is white
        board = tic_tac_toe.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                policy = mcts1.policy_values(board, net1, mcts_sim_count, temp)
            else:
                policy = mcts2.policy_values(board, net2, mcts_sim_count, temp)

            move = np.where(policy == 1)[0]
            board.play_move(move)

            # board.print()
        net1_score += board.white_score()
        net2_score += board.black_score()


        # play half the games where the new network is white
        board = tic_tac_toe.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                policy = mcts2.policy_values(board, net2, mcts_sim_count, temp)
            else:
                policy = mcts1.policy_values(board, net1, mcts_sim_count, temp)

            move = np.where(policy == 1)[0]
            board.play_move(move)

        net1_score += board.black_score()
        net2_score += board.white_score()

    return net1_score, net2_score


def net_vs_minimax(net, game_count, mcts_sim_count, c_puct, temp):
    """
    lets the alpha zero network play against a minimax player
    :param net:             alpha zero network
    :param game_count:      total games to play
    :param mcts_sim_count   number of monte carlo simulations
    :param c_puct           constant that controls the exploration
    :param temp             the temperature
    :return:                score of network
    """
    minimax.fill_state_dict()  # ensure that the state dict is filled

    half_game_count = int(game_count / 2)
    score = 0

    mcts_net = mcts.MCTS(c_puct)
    for _ in range(half_game_count):
        # play half the games where the net is white
        board = tic_tac_toe.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                policy = mcts_net.policy_values(board, net, mcts_sim_count, temp)
                move = np.where(policy == 1)[0]
            else:
                move = board.minimax_move()

            board.play_move(move)

            # board.print()
        score += board.white_score()


        # play half the games where the net is black
        board = tic_tac_toe.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                move = board.minimax_move()
            else:
                policy = mcts_net.policy_values(board, net, mcts_sim_count, temp)
                move = np.where(policy == 1)[0]

            board.play_move(move)

        score += board.black_score()

    return score
