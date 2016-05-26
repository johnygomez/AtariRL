import sys
import os
sys.path.append(os.path.abspath('../lib'))
from AleAgent import AleAgent
from ImgProc.PongProcessing import PongProcessing as proc

if __name__ == "__main__":
  agent = AleAgent(proc, game_rom = 'Pong.bin', encoder_model = 'encoder_model_100.json', encoder_weights = 'encoder_weights_100.h5')
  agent.train()