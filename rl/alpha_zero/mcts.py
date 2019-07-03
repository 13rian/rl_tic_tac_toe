import math

import numpy as np
import torch

from game.globals import CONST, Globals


class MCTS:
    """
    handles the monte-carlo tree search
    """
    
    def __init__(self, board, c_puct):
        self.board = board.clone()  # represents the game
        self.c_puct = c_puct        # a tuneable hyperparameter. The larger this number, the more the model explores
        
        self.P = {}                 # holds the policies for a game state, key: s, value: policy
        self.Q = {}                 # action value, key; (s,a)
        self.N_s = {}               # number of times the state s was visited, key: s
        self.N_sa = {}              # number of times action a was chosen in state s, key: (s,a)
        
        
    
    def policy_values(self, net, mc_sim_count, temp):
        """
        executes mc_sim_count number of monte-carlo simulations to obtain the probability
        vector of the current game position
        :param net:              neural network that approximates the policy and the value
        :param mc_sim_count:     number of monte-carlo simulations to perform
        :param temp:             the temperature, determines the degree of exploration
                                 temp = 0 means that we only pick the best move
                                 temp = 1 means that we pick the move proportional to the count the state was visited  
        :return:                 policy where the probability of an action is proportional to 
                                 N_sa**(1/temp)
        """
               
        # perform the tree search
        for _ in range(mc_sim_count):
            self.tree_search(net)

        s = self.board.state_number()
        counts = [self.N_sa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(CONST.NN_ACTION_SIZE)]

        # in order to learn something set the probabilities of the best action to 1 and all other action to 0
        if temp == 0:
            action = np.argmax(counts)
            probs = [0]*CONST.NN_ACTION_SIZE
            probs[action] = 1
            return probs
        
        else:
            counts = [c**(1./temp) for c in counts]
            probs = [c / float(sum(counts)) for c in counts]
            return probs    
    
        
        
    def tree_search(self, net):
        """
        Performs one iteration of the monte-carlo tree search.
        The method is recursively called until a leaf node is found. This is a game
        state from wich no simulation (playout) has yet been initiated. If the leaf note
        is a terminal state, the reward is returned. If the leaf node is not a terminal node
        the value is estimated with the neural network. 
        The move (action) with the highest upper confidence bound is chosen. The fewer a move was
        chosen in a certain position the higher his upper confidence bound.
        
        The method returns the estimated value of the current game state. The sign of the value
        is flipped because the value of the game for the other player is the negative value of
        the state of the current player          
        :param net:       neural network that approximates the policy and the value
        :return: 
        """
    
        # check if the game is terminal    
        if self.board.terminal:
            return -self.board.reward()
    
        # check if we are on a leaf node (state form which no simulation was played so far)
        s = self.board.state_number()
        if s not in self.P:
            batch, _ = self.board.bit_board_representation()  
            batch = torch.Tensor(batch).to(Globals.device)
            self.P[s], v = net(batch)
            
            # ensure that the summed probability of all valid moves is 1
            legal_moves = np.array(self.board.legal_moves)
            legal_move_indices = np.zeros(len(legal_moves))
            legal_move_indices[legal_moves] = 1
            legal_move_indices = torch.Tensor(legal_move_indices).to(Globals.device)
            self.P[s] = self.P[s] * legal_move_indices
            total_prob = torch.sum(self.P[s], 1).item()
            if total_prob > 0:
                self.P[s] /= total_prob    # normalize the probabilities
            
            else:
                # the network did not choose any legal move, make all moves equally probable
                print("warning: total probabilities estimated by the network for all legal moves is smaller than 0") 
                self.P[s][legal_moves] = 1 / len(legal_moves)
            
            self.N_s[s] = 0
            return -v
      
        # choose the action with the highest upper confidence bound
        max_ucb = -float("inf")
        action = -1
        for a in self.board.legal_moves:
            if (s,a) in self.Q:
                u = self.Q[(s,a)] + self.c_puct*self.P[s][a]*math.sqrt(self.N_s[s]) / (1+self.N_sa[(s,a)])
            else:
                u = 0
            

            if u > max_ucb:
                max_ucb = u
                action = a
        
        a = action
        self.board.play_move(a)
        v = self.tree_search(net)
        
        
        # update the Q and N values
        if (s,a) in self.Qsa:
            self.Q[(s,a)] = (self.N_sa[(s,a)]*self.Q[(s,a)] + v) / (self.N_sa[(s,a)] + 1)
            self.N_sa[(s,a)] += 1
        else:   
            self.Q[(s,a)] = v
            self.N_sa[(s,a)] = 1
        
        self.N_s[s] += 1
        return -v
    
    
    
    
    