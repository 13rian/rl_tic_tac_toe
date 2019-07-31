from utils import utils

from game import tournament
from game.globals import CONST

import logging


# initialize the logger
# The logger
utils.init_logger(logging.DEBUG, file_name="log/app.log")
logger = logging.getLogger('Tests')


# play random vs random
game_count = 1000
player1 = tournament.RandomPlayer()
player2 = tournament.RandomPlayer()
white_score = tournament.play_one_color(game_count, player1, CONST.WHITE, player2)
logger.info("white score for random vs random: {}".format(white_score))

black_score = tournament.play_one_color(game_count, player1, CONST.BLACK, player2)
logger.info("black score for random vs random: {}".format(black_score))


# play minimax vs minimax to check if the score is 0.5
game_count = 100
player1 = tournament.MinimaxPlayer()
player2 = tournament.MinimaxPlayer()
player1_score = tournament.play_match(game_count, player1, player2)
logger.info("minimax vs minimax score: {}".format(player1_score))


# play random vs minimax
game_count = 1000
player1 = tournament.MinimaxPlayer()
player2 = tournament.RandomPlayer()
white_score = tournament.play_one_color(game_count, player1, CONST.WHITE, player2)
logger.info("white score for minimax vs random: {}".format(white_score))

black_score = tournament.play_one_color(game_count, player1, CONST.BLACK, player2)
logger.info("black score for minimax vs random: {}".format(black_score))







