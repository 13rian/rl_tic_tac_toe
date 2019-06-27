import numpy as np

class CONST:
	EMPTY = 0 			# no disk
	WHITE = 1			# white disk (cross, x)
	BLACK = 2 			# black disk (circle, o)

	NN_INPUT_SIZE = 18  # size of the neural network input

	WHITE_MOVE = 0		# white's move constant
	BLACK_MOVE = 1 		# black's move constant


class Globals:
	device = None 		# the pytorch device that is used for training
