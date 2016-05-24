import sys
import os
sys.path.append(os.path.abspath('../lib'))
from AleAgent import AleAgent

if __name__ == "__main__":
  agent = AleAgent(encoder_model = 'encoder_model.json', encoder_weights = 'encoder_weights.h5')
  agent.train()