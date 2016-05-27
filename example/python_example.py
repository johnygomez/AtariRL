import sys
from random import randrange
from ale_python_interface import ALEInterface



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

# Get the list of legal actions
legal_actions = ale.getMinimalActionSet()
print legal_actions
# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    print 'Reward acquired: ', reward
    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()
