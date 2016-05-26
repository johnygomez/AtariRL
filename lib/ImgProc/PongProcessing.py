import numpy as np
from Autoencoder.Encoder import Encoder

class PongProcessing:
  def __init__(self):
    self.pooling_data = np.zeros((40,40), dtype=np.uint8)


  def __squareLoop(self, I, J, trimmed_data):
  	for i in range(I, I+4):
  		for j in range(J, J+4):
  			if(trimmed_data[i,j] != 87):
  				self.pooling_data[I/4,J/4] = 1
  				return


  def process(self, screen_data): 
    trimmed_data = np.delete(screen_data, np.s_[195:], 0)
    for i in range(0, 160):
    	trimmed_data[194,i] = 87
    
    trimmed_data = np.delete(trimmed_data, np.s_[0:35], 0)
    self.pooling_data = np.zeros((40,40), dtype=np.uint8)
    for I in range(0,160,4):
     	for J in range(0,160,4):
        	self.__squareLoop(I,J,trimmed_data)

    return np.array(self.pooling_data.reshape(1,1600))