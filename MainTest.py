from utils import utils

import numpy as np
import logging
from rl.alpha_zero import mcts, alpha_zero_learning
from game.tic_tac_toe import BitBoard

# initialize the logger
# The logger
utils.init_logger(logging.DEBUG, file_name="log/app.log")
logger = logging.getLogger('OthelloTrain')

print("test")
a = 5
print("hello")


a = 6
b = 7
c = 0.5*(a + b) * (a + b + 1) + b
print(int(c))

a = np.zeros(5)
i = np.array([0,2])
a[i] = 1
print(a)

board = BitBoard()
mcts = mcts.MCTS(board, 4)
net = alpha_zero_learning.Network(0.001)
mcts.policy_values(net, 5, 0.5)








