import time
import logging

from game.globals import CONST
from game import tic_tac_toe

logger = logging.getLogger('Minimax')

state_dict = {}     # holds all states of the game, key: state_number, value: white score


def minimax(board, player, fill_state_dict = False):
    """
    optimally solves tic tac toe in every board position. this is an implementation of the
    minimax algorithm. it fills the state dict with the seen states
    :param board:       the tic tac toe board
    :param player:      the current player
    :return:            the score of the current player
                         1: win
                         0: draw
                        -1: loss
    """
    if board.terminal:
        reward = board.reward()
        if player == CONST.BLACK:
            reward = -reward
        return reward

    move = -1
    score = -2

    for a in board.legal_moves:
        board_clone = board.clone()
        current_player = board_clone.player
        board_clone.play_move(a)      # try out the move
        move_score = -minimax(board_clone, board_clone.player, fill_state_dict)      # get the score for the opponent

        # fill the state dict
        if fill_state_dict:
            white_score = move_score if current_player == CONST.WHITE else -move_score
            state_number = board_clone.state_number()
            state_dict[state_number] = white_score

        if move_score > score:
            score = move_score
            move = a

    if move == -1:
        return 0

    return score


def fill_state_dict():
    """
    fills the state dict with the white score values
    :return:
    """
    if len(state_dict) > 0:
        return

    logger.debug("start to fill the minimax state dict")
    start_time = time.time()
    board = tic_tac_toe.BitBoard()

    # fill in the first state
    state = board.state_number()

    # go through the whole game
    score = minimax(board, board.player, True)
    state_dict[state] = score
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("elapsed time to fill the state dict: {}".format(elapsed_time))
    logger.debug("size of the state dict: {}".format(len(state_dict)))
