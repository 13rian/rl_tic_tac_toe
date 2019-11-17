import random
import copy

import numpy as np
import torch

from utils import utils
from game.globals import CONST
from game.globals import Globals
from rl.alpha_zero import mcts
from game import minimax


class BitBoard:
    """
    each player gets a separate board representation
    0  1  2
    3  4  5
    6  7  8
    if a stone is set for a player the bit string will have a 1 on the correct position
    a move is defined by a number, e.g. 4 (this represents setting a stone on the board position 4)
    """

    def __init__(self):
        self.white_player = 0
        self.black_player = 0
        
        self.player = CONST.WHITE                   # disk of the player to move
        self.terminal = False                       # is the game finished
        self.score = 0                              # -1 if black wins, 0 if it is a tie and 1 if white wins
        self.legal_moves = []                       # holds all legal moves of the current board position
                
        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()
        
        
    def clone(self):
        """
        returns a new board with the same state
        :return:
        """
        board = copy.deepcopy(self)
        return board
        
    
    #################################################################################################
    #                                 board representation                                          #
    #################################################################################################

    def from_board_matrix(self, board):
        """
        creates the bit board from the passed board representation
        :param board:   game represented as one board
        :return:
        """

        white_board = board == CONST.WHITE
        white_board = white_board.astype(int)
        self.white_player = self.board_to_int(white_board)
        
        black_board = board == CONST.BLACK
        black_board = black_board.astype(int)
        self.black_player = self.board_to_int(black_board)
        
        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()
        
        # check the game states
        self.swap_players()
        self.check_win()
        self.swap_players()
    

    def print(self):
        """
        prints the current board configuration
        :return:
        """

        # create the board representation form the bit strings
        print(self.get_board_matrix())


    def get_board_matrix(self):
        """
        :return:  human readable game board representation
        """

        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        board = np.add(white_board * CONST.WHITE, black_board * CONST.BLACK)
        return board
    

    def bit_board_representation(self):
        """
        creates a vector form the current game state, 1xnn_input_size,
        the first 9 values are white followed by the black position
        :return:
        """

        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        bit_board = np.stack((white_board, black_board), axis=0)
        player = CONST.WHITE_MOVE if self.player == CONST.WHITE else CONST.BLACK_MOVE
        bit_board = bit_board.reshape((1, CONST.NN_INPUT_SIZE))
        return bit_board, player
    
    
    def white_perspective(self):
        """
        returns the board from the white perspective. If it is white's move the normal board representation is returned.
        If it is black's move the white and the black pieces are swapped.
        :return:
        """
        
        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        if self.player == CONST.WHITE:
            bit_board = np.stack((white_board, black_board), axis=0)
            player = CONST.WHITE_MOVE
        else:
            bit_board = np.stack((black_board, white_board), axis=0)
            player = CONST.BLACK_MOVE
        
        bit_board = bit_board.reshape((1, CONST.NN_INPUT_SIZE))
        return bit_board, player
    

    def int_to_board(self, number):
        """
        creates the 3x3 bitmask that is represented by the passed integer
        :param number:      move on the board
        :return:            x3 matrix representing the board
        """

        number = (number & 7) + ((number & 56) << 5) + ((number & 448) << 10)    
        byte_arr = np.array([number], dtype=np.uint32).view(np.uint8)
        board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[0:3, ::-1][:, 0:3]
        return board_mask
        

    def board_to_int(self, mask):
        """
        converts the passed board mask (3x3) to an integer
        :param mask:    binary board representation 3x3
        :return:        integer representing the passed board
        """
        bit_arr = np.reshape(mask, -1).astype(np.uint32)
        number = bit_arr.dot(1 << np.arange(bit_arr.size, dtype=np.uint32))
        return int(number)
    

    def move_to_board_mask(self, move):
        """
        :param move:    integer defining a move on the board
        :return:        the move represented as a mask on the 3x3 board
        """

        mask = 1 << move
        board_mask = self.int_to_board(mask)
        return board_mask
    
    
    def state_id(self):
        """
        uses the cantor pairing function to create one unique id for the state form the two integers representing the
        board state
        """
        state = "{}_{}".format(self.white_player, self.black_player)
        return state
    
    
    
    #################################################################################################
    #                                 game play management                                          #
    #################################################################################################

    def play_move(self, move):
        """
        plays the passed move on the board
        :param move:    integer that defines the position to set the stone
        :return:
        """
        # if move not in self.legal_moves:
        #     print("move not in list")

        # set the token
        if self.player == CONST.WHITE:
            self.white_player = self.white_player + (1 << move)
        else:
            self.black_player = self.black_player + (1 << move)
      
        # check if the player won
        self.check_win()
        
        # swap the active player and calculate the legal moves
        self.swap_players()
        self.__calc_legal_moves__()
        

    def check_win(self):
        """
        checks if the current player has won the game
        :return:
        """

        if self.three_in_a_row(self.player):
            self.terminal = True
            self.score = 1 if self.player == CONST.WHITE else -1        
        

    def swap_players(self):
        self.player = CONST.WHITE if self.player == CONST.BLACK else CONST.BLACK
        

    def random_move(self):
        if len(self.legal_moves) > 0:
            index = random.randint(0, len(self.legal_moves) - 1)
            return self.legal_moves[index]
        else:
            return None
            

    def __calc_legal_moves__(self):
        # define the mask with all legal moves
        move_mask = utils.bit_not(self.white_player ^ self.black_player, 9)    # this is basically an xnor (only 1 if both are 0)
        
        self.legal_moves = []
        for move in range(9):
            if (1 << move) & move_mask > 0:
                self.legal_moves.append(move)
                
        # if there are no legal moves the game is drawn
        if len(self.legal_moves) == 0:
            self.terminal = True
     

    def three_in_a_row(self, player):
        """
        checks if the passed player has a row of three
        :param player:      the player for which 3 in a row is checked
        :return:
        """

        board = self.white_player if player == CONST.WHITE else self.black_player
        
        # horizontal check
        if board & 7 == 7 or board & 56 == 56 or board & 448 == 448:
            return True 
        
        # vertical check
        if board & 73 == 73 or board & 146 == 146 or board & 292 == 292:
            return True

        # diagonal check /
        if board & 84 == 84: 
            return True
        
        # diagonal check \
        if board & 273 == 273:  
            return True
        
        # nothing found
        return False
    

    def not_terminal_int(self):
        return 0 if self.terminal else 1


    def set_player_white(self):
        self.player = CONST.WHITE
        

    def set_player_black(self):
        self.player = CONST.BLACK
       
       
       
       
    #################################################################################################
    #                               network training methods                                        #
    #################################################################################################
    def reward(self):
        """
        :return:    -1 if black has won
                    0 if the game is drawn or the game is still running
                    1 if white has won
        """

        if not self.terminal:
            return 0        
        else:
            return self.score


    def white_score(self):
        reward = self.reward()
        return (reward + 1) / 2


    def black_score(self):
        reward = self.reward()
        return (-reward + 1) / 2
        

    def all_after_states(self):
        """
        returns all after states (board positions after all legal moves were played)
        :return:  nxnn_input_size matrix where n is the number of legal moves and nn_input_size the bit board representation
                  player vector, 1: if it is whites move, 0 if it is blacks move
        """

        legal_move_count = len(self.legal_moves)
        if legal_move_count == 0:
            return None
        
        after_states = np.empty((legal_move_count, CONST.NN_INPUT_SIZE))
        player = np.empty((legal_move_count, 1))
        for i, move in enumerate(self.legal_moves):
            # create a board with the next move played
            if self.player == CONST.WHITE:
                white_player = self.white_player + (1 << move)
                black_player = self.black_player
            else:
                white_player = self.white_player
                black_player = self.black_player + (1 << move)
            
            # create the bit board
            white_board = self.int_to_board(white_player)
            black_board = self.int_to_board(black_player)
            bit_board = np.stack((white_board, black_board), axis=0).reshape((1, CONST.NN_INPUT_SIZE))
            after_states[i, :] = bit_board
            player[i, 0] = CONST.WHITE_MOVE if self.player == CONST.BLACK else CONST.BLACK_MOVE
        
        return after_states, player
        

    def greedy_value_move(self, net):
        """
        returns the legal move with the highest action value. The ides is that q(s,a) = argmax v(s')
        where s' is the after state (state after the move was played)
        Both players will make an optimal move according to the maximal action value defined by the
        approximated action value function
        :param net:     the network that approximates the value function
        :return:        - the best move according to the networks value function
                        - the best value
        """

        after_states, players = self.all_after_states()
        after_states = torch.Tensor(after_states).to(Globals.device)
        players = torch.Tensor(players).to(Globals.device)
        values = net(after_states, players)
        
        # pick the maximal value move for white and the minimal value move for black
        if self.player == CONST.WHITE:
            value, index = values.max(0)
        else:
            value, index = values.min(0)
        
        greedy_move = self.legal_moves[index]
        return greedy_move, value



    def greedy_action_move(self, net):
        """
        returns the legal move with the highest action value max Q(s, a)
        Both players will make an optimal move according to the maximal action value defined by the
        approximated action value function
        :param net:     the network that approximates the action value function
        :return:        - the best move according to the networks action value function
                        - the best action value
        """

        state, _ = self.bit_board_representation()
        state = torch.Tensor(state).to(Globals.device)
        q_values = net(state)

        # pick the move with the maximal action value for white and the minimal action value for black
        if self.player == CONST.WHITE:
            legal_q_values = q_values[0, 0:9][self.legal_moves]
            value, index = legal_q_values.max(0)
        else:
            legal_q_values = q_values[0, 9:18][self.legal_moves]
            value, index = legal_q_values.min(0)

        greedy_move = self.legal_moves[index]
        return greedy_move, value


    def minimax_move(self):
        """
        returns the optimal minimax move, if there are more than one optimal moves, a random one is
        picked
        :return:
        """

        # get the white score for all legal moves
        score_list = np.empty(len(self.legal_moves))
        for idx, move in enumerate(self.legal_moves):
            board_clone = self.clone()
            board_clone.play_move(move)
            state = board_clone.state_id()
            white_score = minimax.state_dict.get(state)
            score_list[idx] = white_score

        # find the indices of the max score for white and the min score for black
        if self.player == CONST.WHITE:
            move_indices = np.argwhere(score_list == np.amax(score_list))
        else:
            move_indices = np.argwhere(score_list == np.amin(score_list))

        move_indices = move_indices.squeeze(axis=1)
        best_moves = np.array(self.legal_moves)[move_indices]
        best_move = np.random.choice(best_moves, 1)
        return int(best_move)
