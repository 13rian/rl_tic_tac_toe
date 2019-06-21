import numpy as np

class CONST:
	EMPTY = 0 			# no disk
	WHITE = 1			# white disk (cross, x)
	BLACK = 2 			# black disk (circle, o)

	NN_INPUT_SIZE = 27  # size of the neural network input

	WHITE_MOVE_VEC = np.zeros((3, 3)) 	# plane if it is white'S move
	BLACK_MOVE_VEC = np.ones((3, 3)) 	# plane if it is black'S move


class Globals:
	device = None 		# the pytorch device that is used for training
