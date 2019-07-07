from utils import utils

import numpy as np
import torch
import logging
from rl.alpha_zero import mcts, alpha_zero_learning
from game.tic_tac_toe import BitBoard
import time

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


tuple_list = [(1.0,1.1,1.2), (2.0,2.1,2.2), (3.0,3.1,3.2)]
sample_ids = [0,1,2]
a, b, c = list(zip(*[tuple_list[i] for i in sample_ids]))


# board = BitBoard()
# mcts_test = mcts.MCTS(4)
# net = alpha_zero_learning.Network(0.001)
# 
# start = time.time()
# policy = mcts_test.policy_values(board, net, 80, 0)
# end = time.time()
# print("time: {}, policy: {}".format(end-start, policy))


# mat = np.array([
#     [0, 0, 1],
#     [1, 2, 2],
#     [1, 2, 1]
# ])
# mcts_test = mcts.MCTS(4)
# board = BitBoard()
# board.from_board_matrix(mat)
# board.set_player_black()
# net = alpha_zero_learning.Network(0.001)
# policy = mcts_test.policy_values(board, net, 80, 1)
# print(policy)



# mat = np.array([
#     [0, 0, 1],
#     [1, 2, 2],
#     [1, 2, 1]
# ])
mat = np.array([
    [0, 0, 0],
    [1, 2, 0],
    [1, 0, 2]
])
board = BitBoard()
board.from_board_matrix(mat)
board.set_player_black()

net = alpha_zero_learning.Network(0.001)
net = torch.load("rl/alpha_zero/ticTacToeSelfPlay.pt")

state, player = board.white_perspective()
batch = torch.tensor(state, dtype=torch.float32)
policy, value = net(batch)
print(policy)
print(value)








