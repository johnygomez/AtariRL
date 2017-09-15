import numpy as np
import matplotlib.pyplot as plt
from Autoencoder.Encoder import Encoder


##
#   Goal of this class is to trim and reduce the dimension of input image (game UI).
#   Processing of every game is unfortunately very different, so it's hard apply any kind of abstraction.
class PongProcessing:
    def __init__(self):
        self.pooling_data = np.zeros((40, 31), dtype=np.uint8)

    ##
    # Detects ball or paddle in pooled square
    #
    # @param I row
    # @param J col
    # @param trimmed_data 4x4 square trimmed from input image
    def __squareLoop(self, I, J, trimmed_data):
        for i in range(I, min(I+4, len(trimmed_data))):
            for j in range(J, min(J+4, len(trimmed_data[0]))):
                if(trimmed_data[i, j] != 87):
                    self.pooling_data[I/4, J/4] = 1

    ##
    # Trim input image and reduce dimension using MaxPooling
    def process(self, screen_data):
        trimmed_data = np.delete(screen_data, np.s_[194:], 0)
        trimmed_data = np.delete(trimmed_data, np.s_[0:35], 0)
        trimmed_data = np.delete(trimmed_data, np.s_[0:20], 1)
        trimmed_data = np.delete(trimmed_data, np.s_[124:], 1)

        for I in range(0, len(trimmed_data), 4):
            for J in range(0, len(trimmed_data[0]), 4):
                self.__squareLoop(I, J, trimmed_data)
                return np.array(self.pooling_data.reshape(1, len(self.pooling_data)*len(self.pooling_data[0])))
