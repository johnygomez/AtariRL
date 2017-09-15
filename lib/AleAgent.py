import sys
import pygame
import numpy as np
from ale_python_interface import ALEInterface
from Autoencoder.Encoder import Encoder
from NFQ.NFQ import NFQ


##
# Main Class of the program initializing vizual UI of the game and also a processing and training phase
class AleAgent:
    ##
    # @param processing_cls Class for processing game visual unput
    def __init__(self, processing_cls, game_rom=None, encoder_model=None, encoder_weights=None, NFQ_model=None, NFQ_weights=None):
        assert game_rom is not None
        self.game = ALEInterface()
        if encoder_weights is not None and encoder_model is not None:
            self.encoder = Encoder(path_to_model=encoder_model, path_to_weights=encoder_weights)
        else:
            self.encoder = Encoder()

        self.processor = processing_cls()

        # Get & Set the desired settings
        self.game.setInt('random_seed', 0)
        self.game.setInt('frame_skip', 4)

        # Set USE_SDL to true to display the screen. ALE must be compilied
        # with SDL enabled for this to work. On OSX, pygame init is used to
        # proxy-call SDL_main.
        USE_SDL = True

        if USE_SDL:
            if sys.platform == 'darwin':
                pygame.init()
                self.game.setBool('sound', False)   # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.game.setBool('sound', False)   # no sound

            self.game.setBool('display_screen', True)

        # Load the ROM file
        self.game.loadROM(game_rom)

        # Get the list of legal actions
        self.legal_actions = self.game.getLegalActionSet()

        # Get actions applicable in current game
        self.minimal_actions = self.game.getMinimalActionSet()

        if NFQ_model is not None and NFQ_weights is not None:
            self.NFQ = NFQ(
                self.encoder.out_dim,
                len(self.minimal_actions),
                model_path=NFQ_model,
                weights_path=NFQ_weights
            )
        else:
            self.NFQ = NFQ(self.encoder.out_dim, len(self.minimal_actions))

        (self.screen_width, self.screen_height) = self.game.getScreenDims()
        self.screen_data = np.zeros(
            (self.screen_height, self.screen_width),
            dtype=np.uint8
        )

    ##
    # Initialize the reinforcement learning
    def train(self, num_of_episodes=1500, eps=0.995, key_binding=None):
        pygame.init()
        for episode in xrange(num_of_episodes):
            total_reward = 0
            moves = 0
            hits = 0
            print 'Starting episode: ', episode+1

            if key_binding:
                eps = 0.05
            else:
                eps -= 2/num_of_episodes

            self.game.getScreenGrayscale(self.screen_data)
            pooled_data = self.processor.process(self.screen_data)
            next_state = self.encoder.encode(pooled_data)
            while not self.game.game_over():
                current_state = next_state
                x = None

                if key_binding:
                    key_pressed = pygame.key.get_pressed()
                    x = key_binding(key_pressed)

                if x is None:
                    r = np.random.rand()
                    if r < eps:
                        x = np.random.randint(self.minimal_actions.size)
                    else:
                        x = self.NFQ.predict_action(current_state)

                a = self.minimal_actions[x]
                # Apply an action and get the resulting reward
                reward = self.game.act(a)

                # record only every 3 frames
                # if not moves % 3:
                self.game.getScreenGrayscale(self.screen_data)
                pooled_data = self.processor.process(self.screen_data)
                next_state = self.encoder.encode(pooled_data)
                transition = np.append(current_state, x)
                transition = np.append(transition, next_state)
                transition = np.append(transition, reward)
                self.NFQ.add_transition(transition)

                total_reward += reward
                if reward > 0:
                    hits += 1

                moves += 1
                if eps > 0.1:
                    eps -= 0.00001
            # end while

            print 'Epsilon: ', eps
            print 'Episode', episode+1, 'ended with score:', total_reward
            print 'Hits: ', hits
            self.game.reset_game()
            self.NFQ.train()
            hits = 0
            moves = 0
            self.NFQ.save_net()
        # end for

    ##
    # Play the game!
    def play(self):
        total_reward = 0
        moves = 1
        while not self.game.game_over():
            self.game.getScreenGrayscale(self.screen_data)
            pooled_data = self.processor.process(self.screen_data)
            current_state = self.encoder.encode(pooled_data)

            x = self.NFQ.predict_action(current_state)
            a = self.minimal_actions[x]
            reward = self.game.act(a)
            total_reward += reward
            moves += 1

        print 'The game ended with score:', total_reward, ' after: ', moves, ' moves'
