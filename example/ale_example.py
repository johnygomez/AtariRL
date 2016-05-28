import sys
from random import randrange
from ale_python_interface import ALEInterface
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../lib'))
from ImgProc.PongProcessing import PongProcessing as proc
from Autoencoder.Encoder import Encoder
import matplotlib.pyplot as plt



processor = proc()
encoder = Encoder(path_to_model = 'encoder_v2_model.json', path_to_weights = 'encoder_v2_weights.h5')

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
    ale.setBool('sound', False)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM('Pong.bin')

# Get the list of legal actionsl;
legal_actions = ale.getLegalActionSet()

(screen_width, screen_height) = ale.getScreenDims()
#screen_data = np.zeros(screen_width*screen_height, dtype=np.uint32)
screen_data = np.zeros((screen_height,screen_width), dtype=np.uint8)
pooling_data = np.zeros((40,31), dtype=np.uint8)


# Play 10 episodes
for episode in xrange(20):
  total_reward = 0
  i = 0
  while not ale.game_over():
    i = i + 1
    if i % 20 == 0:
      ale.getScreenGrayscale(screen_data)
      pooled_data = processor.process(screen_data)
      encoded_data = encoder.draw(pooled_data)
      plt.figure(figsize=(1, 1), dpi=40)
      plt.imshow(encoded_data.reshape(40, 31))
      plt.show()

      
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    #print 'Reward acquired: ', reward
    total_reward += reward
  
  

  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
