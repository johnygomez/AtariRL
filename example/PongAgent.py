import sys
import os
sys.path.append(os.path.abspath('../lib'))
from AleAgent import AleAgent
from ImgProc.PongProcessing import PongProcessing as proc
import pygame

def key_binding(key_pressed):
  if key_pressed[pygame.K_RIGHT]:
    return 1
  elif key_binding and key_pressed[pygame.K_LEFT]:
    return 2
  elif key_binding and key_pressed[pygame.K_DOWN]:
    return 0
  else:
    return None


if __name__ == "__main__":
  agent = AleAgent(proc, game_rom = 'Pong.bin', encoder_model = 'encoder_model.json', encoder_weights = 'encoder_weights.h5', NFQ_model = 'nfq_model.json', NFQ_weights = 'nfq_weights.h5')
  pygame.init()
  agent.train(key_binding = key_binding)