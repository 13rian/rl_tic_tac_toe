from utils import utils

from game import tic_tac_toe
from game.globals import CONST

import numpy as np
import torch
import logging
from rl.alpha_zero import mcts, alpha_zero_learning
from game.tic_tac_toe import BitBoard
from game import minimax
import time

# initialize the logger
# The logger
utils.init_logger(logging.DEBUG, file_name="log/app.log")
logger = logging.getLogger('OthelloTrain')


score = tic_tac_toe.play_minimax_vs_random(1, CONST.WHITE)

start_time = time.time()
score_as_white = tic_tac_toe.play_minimax_vs_random(100, CONST.WHITE)
score_as_black = tic_tac_toe.play_minimax_vs_random(100, CONST.BLACK)
end_time = time.time()
elapsed_time = end_time - start_time

print("score as white: ", score_as_white)
print("score as black: ", score_as_black)
print("elapsed_time: ", elapsed_time)

start_time = time.time()
white_score = tic_tac_toe.play_minimax_vs_minimax(100)
end_time = time.time()
elapsed_time = end_time - start_time

print("minimax vs minimax: ", white_score)
print("elapsed_time: ", elapsed_time)





