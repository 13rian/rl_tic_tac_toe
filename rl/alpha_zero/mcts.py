import math

import numpy as np
import torch

from game.globals import CONST, Globals


class MCTS:
    """
    handles the monte-carlo tree search
    """
    
    def __init__(self, c_puct):
        self.c_puct = c_puct        # a tuneable hyperparameter. The larger this number, the more the model explores
        
        self.P = {}                 # holds the policies for a game state, key: s, value: policy
        self.Q = {}                 # action value, key; (s,a)
        self.N_s = {}               # number of times the state s was visited, key: s
        self.N_sa = {}              # number of times action a was chosen in state s, key: (s,a)
        
        
    
    def policy_values(self, board, net, mc_sim_count, temp):
        """
        executes mc_sim_count number of monte-carlo simulations to obtain the probability
        vector of the current game position
        :param board:            the game board
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
            sim_board = board.clone()
            self.tree_search(sim_board, net)

        s = board.state_number()
        counts = [self.N_sa[(s, a)] if (s, a) in self.N_sa else 0 for a in range(CONST.NN_POLICY_SIZE)]

        # in order to learn something set the probabilities of the best action to 1 and all other action to 0
        if temp == 0:
            action = np.argmax(counts)
            probs = [0]*CONST.NN_POLICY_SIZE
            probs[action] = 1
            return np.array(probs)
        
        else:
            counts = [c**(1./temp) for c in counts]
            probs = [c / float(sum(counts)) for c in counts]
            return np.array(probs)    
    
        
        
    def tree_search(self, board, net):
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
        :param board:     represents the game
        :return: 
        """
    
        # check if the game is terminal    
        if board.terminal:
            return board.reward()
    
        # check if we are on a leaf node (state form which no simulation was played so far)
        s = board.state_number()
        player = board.player
        if s not in self.P:  
            batch, _ = board.white_perspective()
            batch = torch.Tensor(batch).to(Globals.device)
            self.P[s], v = net(batch)
            self.P[s] = self.P[s].detach().squeeze().numpy()
            v = v.item()
            
            # ensure that the summed probability of all valid moves is 1
            legal_moves = np.array(board.legal_moves)
            legal_move_indices = np.zeros(CONST.NN_POLICY_SIZE)
            legal_move_indices[legal_moves] = 1
            self.P[s] = self.P[s] * legal_move_indices
            total_prob = np.sum(self.P[s])
            if total_prob > 0:
                self.P[s] /= total_prob    # normalize the probabilities
            
            else:
                # the network did not choose any legal move, make all moves equally probable
                print("warning: total probabilities estimated by the network for all legal moves is smaller than 0") 
                self.P[s][legal_moves] = 1 / len(legal_moves)
            
            self.N_s[s] = 0
            return v
      
        # choose the action with the highest upper confidence bound
        max_ucb = -float("inf")
        action = -1
        for a in board.legal_moves:
            if (s,a) in self.Q:
                u = self.Q[(s,a)] + self.c_puct*self.P[s][a]*math.sqrt(self.N_s[s]) / (1+self.N_sa[(s,a)])
            else:
                u = self.c_puct*self.P[s][a]*math.sqrt(self.N_s[s] + 1e-8)  # avoid division by 0
            

            if u > max_ucb:
                max_ucb = u
                action = a
        
        a = action
        board.play_move(a)
        v = self.tree_search(board, net)
        
        
        # update the Q and N values
        v_true = v
        if player == CONST.BLACK:
            v_true *= -1     # flip the value for the black player since the game is always viewed from the white perspective
        
        if (s,a) in self.Q:
            self.Q[(s,a)] = (self.N_sa[(s,a)]*self.Q[(s,a)] + v_true) / (self.N_sa[(s,a)] + 1)
            self.N_sa[(s,a)] += 1
        else:   
            self.Q[(s,a)] = v_true
            self.N_sa[(s,a)] = 1
        
        self.N_s[s] += 1
        return v
    
    
    
    
    