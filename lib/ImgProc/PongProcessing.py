import numpy as np
from Autoencoder.Encoder import Encoder
import matplotlib.pyplot as plt

class PongProcessing:
  def __init__(self):
    self.pooling_data = np.zeros((40,31), dtype=np.uint8)


  def __squareLoop(self, I, J, trimmed_data):
  	for i in range(I, min(I+4,len(trimmed_data))):
  		for j in range(J, min(J+4,len(trimmed_data[0]))):
  			if(trimmed_data[i,j] != 87):
  				self.pooling_data[I/4,J/4] = 1
  				return


  def process(self, screen_data): 
    trimmed_data = np.delete(screen_data, np.s_[194:], 0)
    trimmed_data = np.delete(trimmed_data, np.s_[0:35], 0)
    trimmed_data = np.delete(trimmed_data, np.s_[0:20], 1)
    trimmed_data = np.delete(trimmed_data, np.s_[124:], 1)
    
    #np.savetxt('dat1.txt', trimmed_data, fmt='%2.2f')
    self.pooling_data = np.zeros((40,31), dtype=np.uint8)
    for I in range(0,len(trimmed_data),4):
     	for J in range(0,len(trimmed_data[0]),4):
        	self.__squareLoop(I,J,trimmed_data)
 	#np.savetxt('dat1.txt', self.pooling_data, fmt='%2.2f')
    #plt.figure(figsize=(1, 1), dpi=40)
    #plt.imshow(self.pooling_data)
    #plt.show()
    return np.array(self.pooling_data.reshape(1,len(self.pooling_data)*len(self.pooling_data[0])))