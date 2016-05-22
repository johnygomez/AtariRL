from abc import ABCMeta, abstractmethod
## Provides a default interface for Autoencoder class
#
class Autoencoder:
  __metaclass__ = ABCMeta
  
  ## Training a autoencoder
  #
  # This method must be implemented
  @abstractmethod
  def train(self, pretrained_weights): pass

  ## Encoding an input and providing numpy array of features
  #
  # This method must be implemented
  @abstractmethod
  def encode(self, img): pass

  ## Decoding an input(features) and reconstrucing original input(image)
  #
  # This method must be implemented
  @abstractmethod
  def decode(self, img): pass
