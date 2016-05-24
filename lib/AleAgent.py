import sys
from ale_python_interface import ALEInterface
from Autoencoder import Encoder
from NFQ import NFQ
import numpy as np

class AleAgent:
  def __init__(self, game_rom = None, encoder_model = None, encoder_weights = None, NFQ_model = None, NFQ_weights = None):
    assert game is not None  
    self.game = ALEInterface()
    if encoder_weights is not None and encoder_model is not None:
      self.encoder = Encoder(path_to_model = encoder_model, path_to_weights = encoder_weights)
    else:
      self.encoder = Encoder()

    # Get & Set the desired settings
    self.game.setInt('random_seed', 123)

    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    USE_SDL = True

    if USE_SDL:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.game.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.game.setBool('sound', True)
      self.game.setBool('display_screen', True)

    # Load the ROM file
    self.game.loadROM(game_rom)

    # Get the list of legal actions
    self.legal_actions = self.game.getLegalActionSet()

    # Get actions applicable in current game
    self.minimal_actions = self.game.getMinimalActionSet()

    if NFQ_model is not None and NFQ_weights is not None:
      self.NFQ = NFQ(self.encoder.out_dim, self.minimal_actions, model_path = NFQ_model, weights_path = NFQ_weights)
    else:
      self.NFQ = NFQ(self.encoder.out_dim, self.minimal_actions)

    (self.screen_width,self.screen_height) = self.game.getScreenDims()
    self.screen_data = np.zeros(self.screen_width * self.screen_height, dtype=np.uint32)

  def train(self, num_of_episodes = 100, eps = 1):
    for episode in xrange(num_of_episodes):
      total_reward = 0
      moves = 1
      while not self.game.game_over() and moves < 100:
        self.game.getScreenGrayscale(self.screen_data)
        # TODO processing a zakodovanie autoencoderom
        current_state = self.encoder.encode(self.screen_data)

        r = np.random.rand()
        if r < eps:
          x = np.random.randint(self.minimal_actions.size)
        else:
          x = self.NFQ.predict_action(current_state)

        a = self.minimal_actions[x]
        # Apply an action and get the resulting reward
        reward = self.game.act(a)
        self.game.getScreenGrayscale(self.screen_data)
        
        next_state = self.encoder.encode(self.screen_data)
        transition = np.concatenate((current_state, np.array([x]), next_state, reward))
        self.NFQ.add_transition(transition)
        
        total_reward += reward
        moves += 1
        if eps > 0.1:
          eps -= (1/moves)
      #end while
      print 'Episode', episode, 'ended with score:', total_reward
      self.game.reset_game()
      self.NFQ.train()
      #end for
    self.NFQ.save_net()

  def play(self):
    total_reward = 0
    moves = 1
    while not self.game.game_over():
      self.game.getScreenGrayscale(self.screen_data)
      # TODO processing a zakodovanie autoencoderom
      current_state = self.encoder.encode(self.screen_data)
      x = self.NFQ.predict_action(current_state)
      a = self.minimal_actions[x]
      reward = self.game.act(a)
      total_reward += reward
      moves += 1

    print 'The game ended with score:', total_reward, ' after: ', moves, ' moves'