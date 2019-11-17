import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import tic_tac_toe
from game.globals import Globals, CONST
from game import tournament
from rl.alpha_zero.mcts import MCTS



class Network(nn.Module):
    def __init__(self, learning_rate):
        super(Network, self).__init__()
                
        self.fc1 = nn.Linear(CONST.NN_INPUT_SIZE, 256)       # first fully connected layer
        self.fc2 = nn.Linear(256, 128)                       # second fully connected layer
        self.fc3 = nn.Linear(128, 64)                        # third fully connected layer
        
        # policy head
        self.fc4p = nn.Linear(64, CONST.NN_POLICY_SIZE)      # approximation for the action value function Q(s, a)
        
        # value head
        self.fc4v = nn.Linear(64, 1)                         # approximation for the value function V(s)

        # define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        
        # # randomly initialize all the weights
        # mean = 0
        # std = 0.2
        # nn.init.normal_(self.fc1.weight, mean=mean, std=std)
        # nn.init.normal_(self.fc2.weight, mean=mean, std=std)
        # nn.init.normal_(self.fc3.weight, mean=mean, std=std)
        # nn.init.normal_(self.fc4p.weight, mean=mean, std=std)
        # nn.init.normal_(self.fc4v.weight, mean=mean, std=std)
         
        
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
        self.network = Network(learning_rate)                        # the network for the policy and value prediction

        self.board = tic_tac_toe.BitBoard()                          # tic tac toe board
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences

        # send the network to the configured device
        self.network.to(Globals.device)


    def clear_exp_buffer(self):
        self.experience_buffer.clear()
    

    def play_self_play_games(self, game_count, temp_threshold, alpha_dirich=0):
        """
        plays some games against itself and adds the experience into the experience buffer
        :param game_count:      the number of tames to play
        :param temp_threshold:  up to this move the temp will be temp, after the threshold it will be set to 0
                                plays a game against itself with some exploratory moves in it
        :param alpha_dirich     alpha parameter for the dirichlet noise that is added to the root node
        :return:
        """

        if Globals.pool is None:
            self_play_results = [__self_play_worker__(self.network, self.mcts_sim_count,
                                                      self.c_puct, temp_threshold, self.temp, alpha_dirich, game_count)]
        else:
            games_per_process = int(game_count / Globals.n_pool_processes)
            self_play_results = Globals.pool.starmap(__self_play_worker__,
                                                      zip([self.network] * Globals.n_pool_processes,
                                                          [self.mcts_sim_count] * Globals.n_pool_processes,
                                                          [self.c_puct] * Globals.n_pool_processes,
                                                          [temp_threshold] * Globals.n_pool_processes,
                                                          [self.temp] * Globals.n_pool_processes,
                                                          [alpha_dirich] * Globals.n_pool_processes,
                                                          [games_per_process] * Globals.n_pool_processes))

        # add the training examples to the experience buffer
        tot_moves_played = 0
        for sample in self_play_results:
            state = torch.Tensor(sample[0]).reshape(-1, CONST.NN_INPUT_SIZE)
            policy = torch.Tensor(sample[1]).reshape(-1, CONST.NN_POLICY_SIZE)
            value = torch.Tensor(sample[2]).unsqueeze(1)
            tot_moves_played += state.shape[0]

            self.experience_buffer.add_batch(state, policy, value)

        print("average number of moves played: ", tot_moves_played / game_count)



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
            loss_p, loss_v = self.network.train_step(states, policies, values)
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
        az_player = tournament.AlphaZeroPlayer(self.network, self.c_puct, self.mcts_sim_count, 0)
        random_player = tournament.RandomPlayer()
        score = tournament.play_one_color(game_count, az_player, color, random_player)
        return score

     

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

        
        
    def add_batch(self, states, policies, values):
        """
        adds the multiple experiences to the buffer
        :param states:           the state s_t
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
    :return:                score of network1
    """

    az_player1 = tournament.AlphaZeroPlayer(net1, c_puct, mcts_sim_count, temp)
    az_player2 = tournament.AlphaZeroPlayer(net2, c_puct, mcts_sim_count, temp)
    score1 = tournament.play_match(game_count, az_player1, az_player2)
    return score1



def net_vs_minimax(net, game_count, mcts_sim_count, c_puct, temp, color=None):
    """
    lets the alpha zero network play against a minimax player
    :param net:             alpha zero network
    :param game_count:      total games to play
    :param mcts_sim_count   number of monte carlo simulations
    :param c_puct           constant that controls the exploration
    :param temp             the temperature
    :param color            the color of the network
    :return:                score of network
    """

    az_player = tournament.AlphaZeroPlayer(net, c_puct, mcts_sim_count, temp)
    minimax_player = tournament.MinimaxPlayer()

    if color is None:
        az_score = tournament.play_match(game_count, az_player, minimax_player)
    else:
        az_score = tournament.play_one_color(game_count, az_player, color, minimax_player)

    return az_score


def __self_play_worker__(net, mcts_sim_count, c_puct, temp_threshold, temp, alpha_dirich, game_count):
    """
    plays a number of self play games
    :param net:                 the alpha zero network
    :param mcts_sim_count:      the monte carlo simulation count
    :param c_puct:              constant that controls the exploration
    :param temp_threshold:      up to this move count the temperature will be temp, later it will be 0
    :param temp:                the temperature
    :param alpha_dirich:        dirichlet parameter alpha
    :param game_count:          the number of self-play games to play
    :return:                    state_list, policy_list, value_list
    """

    state_list = []
    policy_list = []
    value_list = []
    position_cache = {}         # faster than a shared position dict

    for i in range(game_count):
        board = tic_tac_toe.BitBoard()
        mcts = MCTS(c_puct)  # reset the search tree

        # reset the players list
        player_list = []

        move_count = 0
        while not board.terminal:
            state, player = board.white_perspective()
            temp = 0 if move_count >= temp_threshold else temp
            policy = mcts.policy_values(board, position_cache, net, mcts_sim_count, temp, alpha_dirich)

            # sample from the policy to determine the move to play
            move = np.random.choice(len(policy), p=policy)
            board.play_move(move)

            # save the training example
            state_list.append(state)
            player_list.append(player)
            policy_list.append(policy)
            move_count += 1

        # calculate the values from the perspective of the player who's move it is
        reward = board.reward()
        for player in player_list:
            value = reward if player == CONST.WHITE_MOVE else -reward
            value_list.append(value)

    return state_list, policy_list, value_list
