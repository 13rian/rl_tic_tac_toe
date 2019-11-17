import matplotlib.pyplot as plt
import torch
import random
import os

from utils import utils

from game.globals import Globals
from rl.alpha_zero import alpha_zero_learning


# set the random seed
random.seed(a=None, version=2)


# define the parameters
network_duel_game_count = 40        # number of games that are played between the old and the new network
mcts_sim_count = 25                 # the number of simulations for the monte-carlo tree search
c_puct = 4                          # the higher this constant the more the mcts explores
network_dir = "networks"            # directory in which the networks are saved

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.device = torch.device('cpu')



# let the networks play against a minimax player
generation_nm = []
avg_score_nm = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)

# get the best network
best_network_path = "{}/{}".format(network_dir, path_list[-1])
best_net = torch.load(best_network_path).to(Globals.device)
for i in range(len(path_list)):
    generation_nm.append(i)
    net_path = "{}/{}".format(network_dir, path_list[i])
    net = torch.load(net_path).to(Globals.device)

    print("play {} against minimax".format(net_path))
    net_score = alpha_zero_learning.net_vs_minimax(net, network_duel_game_count, mcts_sim_count, c_puct, 0)
    avg_score_nm.append(net_score)




# let the different networks play against each other
generation_nn = []
avg_score_nn = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)

# get the best network
best_network_path = "{}/{}".format(network_dir, path_list[-1])
best_net = torch.load(best_network_path).to(Globals.device)
for i in range(len(path_list)):
    generation_nn.append(i)
    net_path = "{}/{}".format(network_dir, path_list[i])
    net = torch.load(net_path).to(Globals.device)

    print("play {} against the best network {}".format(net_path, best_network_path))
    # random_net = alpha_zero_learning.Network(learning_rate)
    net_score = alpha_zero_learning.net_vs_net(net, best_net, network_duel_game_count, mcts_sim_count, c_puct, 0)
    avg_score_nn.append(net_score)



# plot the score of the different generation network against the minimax player
fig1 = plt.figure(1)
plt.plot(generation_nm, avg_score_nm, color="#9ef3f3")
axes = plt.gca()
axes.set_ylim([0, 0.5])
plt.title("Average Score Against Minimax")
plt.xlabel("Generation")
plt.ylabel("Average Score")
fig1.show()


# plot the score of the different generation network against the best network
fig2 = plt.figure(2)
plt.plot(generation_nn, avg_score_nn, color="#9ef3f3")
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title("Average Score Against Best Network")
plt.xlabel("Generation")
plt.ylabel("Average Score")
fig2.show()

plt.show()
