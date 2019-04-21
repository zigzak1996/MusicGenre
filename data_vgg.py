import sys
from utils import *

read_data(sys.argv[1], 660000)
split_for_vgg("data.npy")
