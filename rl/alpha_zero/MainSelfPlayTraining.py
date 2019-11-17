import os
import shutil
import matplotlib.pyplot as plt
import torch
import random
import time
import logging
import multiprocessing as mp

from utils import utils

from game.globals import Globals
from game.globals import CONST
from rl.alpha_zero import alpha_zero_learning

# @utils.profile()
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/tic_tac_toe.log")
    logger = logging.getLogger('Alpha Tic')


    # set the random seed
    random.seed(a=None, version=2)

    # initialize the pool
    Globals.n_pool_processes = 5    # mp.cpu_count()
    Globals.pool = mp.Pool(processes=Globals.n_pool_processes)


    # define the parameters
    epoch_count = 30                    # the number of epochs to train the neural network
    episode_count = 2000                 # the number of games that are self-played in one epoch
    update_count = 200                   # the number the neural net is updated  in one epoch with the experience data
    evaluation_game_count = 300          # the number of games to play against the minimax player
    mcts_sim_count = 25                  # the number of simulations for the monte-carlo tree search
    c_puct = 4                           # the higher this constant the more the mcts explores
    temp = 1                             # the temperature, controls the policy value distribution
    alpha_dirich = 1                     # alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves)
    temp_threshold = 9                   # up to this move the temp will be temp, otherwise 0 (deterministic play)
    learning_rate = 0.001                # the learning rate of the neural network
    batch_size = 128                     # the batch size of the experience buffer for the neural network training
    exp_buffer_size = 10000              # the size of the experience replay buffer
    network_dir = "networks"             # directory in which the networks are saved

    # define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
    Globals.device = torch.device('cpu')


    # create the dirctory to save the networks
    if not os.path.exists(network_dir):
        os.makedirs(network_dir)

    shutil.rmtree(network_dir)
    os.makedirs(network_dir)


    # create the agent
    agent = alpha_zero_learning.Agent(learning_rate, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size)
    torch.save(agent.network, "{}/network_gen_{}.pt".format(network_dir, 0))


    # to plot the fitness
    policy_loss = []
    value_loss = []
    minimax_score_white = []
    minimax_score_black = []


    start_training = time.time()
    for i in range(epoch_count):
        ###### play against a minimax player to see how good the network is
        logger.info("start match against minimax in epoch {}".format(i))
        white_score = alpha_zero_learning.net_vs_minimax(agent.network, evaluation_game_count, mcts_sim_count, c_puct, 0, CONST.WHITE)
        logger.info("white score vs minimax: {}".format(white_score))

        black_score = alpha_zero_learning.net_vs_minimax(agent.network, evaluation_game_count, mcts_sim_count, c_puct, 0, CONST.BLACK)
        logger.info("black score vs minimax: {}".format(black_score))

        minimax_score_white.append(white_score)
        minimax_score_black.append(black_score)


        ###### self play and update: create some game data through self play
        logger.info("start playing games in epoch {}".format(i))
        agent.play_self_play_games(episode_count, temp_threshold, alpha_dirich)


        ###### training, train the training network
        logger.info("start updates in epoch {}".format(i))
        loss_p, loss_v = agent.nn_update(update_count)
        policy_loss.append(loss_p)
        value_loss.append(loss_v)
        print("policy loss: ", loss_p)
        print("value loss: ", loss_v)
        # agent.clear_exp_buffer()            # clear the experience buffer

        # save the current network
        torch.save(agent.network, "{}/network_gen_{}.pt".format(network_dir, i + 1))



    end_training = time.time()
    training_time = end_training - start_training
    logger.info("elapsed time whole training process {} for {} episodes".format(training_time, epoch_count*episode_count))


    # plot the value training loss
    fig1 = plt.figure(1)
    plt.plot(value_loss)
    plt.title("Average Value Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Value Loss")
    fig1.show()

    # plot the training policy loss
    fig2 = plt.figure(2)
    plt.plot(policy_loss)
    plt.title("Average Policy Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    fig2.show()


    # plot the score against the minimax player
    fig3 = plt.figure(3)
    plt.plot(minimax_score_white, label="white")
    plt.plot(minimax_score_black, label="black")
    plt.legend(loc='best')
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.title("Average Score Against Minimax")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    fig3.show()


    plt.show()


if __name__ == '__main__':
    mainTrain()
