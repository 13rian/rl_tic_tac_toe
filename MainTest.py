from utils import utils

import logging

# initialize the logger
# The logger
utils.init_logger(logging.DEBUG, file_name="log/app.log")
logger = logging.getLogger('OthelloTrain')

print("test")
a = 5
print("hello")