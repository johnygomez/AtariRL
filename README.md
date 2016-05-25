# Atari Reinforcement Learning

Aim of this project is to implement simple learning mechanism for Atari 2600 games on ALE Platform ([Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)). As a reinforcement learning technique we use **NFQ Learning** (see next).

### Main Principles of learning:
* First of all, input image of a game is read by ALE interface (lib/ImgProc)
* The input image is encoded using Autoencoder in [lib/Autoencoder/Encoder.py](lib/Autoencoder/Encoder.py)
* The extracted features from encoder (and also reward, action selected.. = transition data) are pushed into the [NFQ module](lib/NFQ/NFQ.py) which saves these transition as a training data for Neural Network learning
* When enough training data are acquired, the NFQ trains the Q-function, which is used as a predictor of the most suitable move

*Find out more about NFQ at [link](http://ml.informatik.uni-freiburg.de/_media/publications/rieecml05.pdf)*.

### Dependencies
* [Pip](https://pypi.python.org/pypi/pip)
* [Numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Keras](http://keras.io/)
* [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment)

### Quick Start
```
make init
```
which installs all dependencies

```
cd example/
python PongAgent.py
```

### Pong game example
To run a simple game example, you first need to get a game binary (you can find one in [example/](example/) folder).
The next step is to train the encoder [lib/Autoencoder/Encoder.py](lib/Autoencoder/Encoder.py) for your game using train method.
When the encoder is train well enough, you can use this pre-trained NN within initialization of [AleAgent](lib/AleAgent.py) after the first required parameter, which is the Game ROM.
[AleAgent](lib/AleAgent.py) contains 2 more methods, which are `train()` and `play()` which first train the Q-function for prediction and then, can play the game using trained network.
