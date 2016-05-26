import sys
from random import randrange
from ale_python_interface import ALEInterface
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../lib'))
from Autoencoder.Encoder import Encoder

def squareLoop(I,J):
  for i in (I, I+4):
    for j in (J, J+4):
      if(trimmed_data2[i,j] != 87):
        pooling_data[I/4,J/4] = 1
        return

encoder = Encoder(pre_trained_model = True, path_to_model = 'encoder_model_200.json', path_to_weights = 'encoder_weights_200.h5')

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM('Pong.bin')

# Get the list of legal actionsl;
legal_actions = ale.getLegalActionSet()

(screen_width, screen_height) = ale.getScreenDims()
#screen_data = np.zeros(screen_width*screen_height, dtype=np.uint32)
screen_data = np.zeros((screen_height,screen_width), dtype=np.uint8)
pooling_data = np.zeros((40,40), dtype=np.uint8)


# Play 10 episodes
for episode in xrange(1):
  total_reward = 0
  i = 0
  #while not ale.game_over():
  for p in xrange(200):
    i = i + 1
    if i % 20 == 0:
      ale.getScreenGrayscale(screen_data)
      trimmed_data = np.delete(screen_data, np.s_[195::], 0)
      trimmed_data2 = np.delete(trimmed_data, np.s_[0:35], 0)
      pooling_data = np.zeros((40,40), dtype=np.uint8)
      for I in range(0,156,4):
        for J in range(0,156,4):
          squareLoop(I,J)
      encoder.encode(np.array(pooling_data.reshape(1,1600)))
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    #print 'Reward acquired: ', reward
    total_reward += reward
  
  

  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
