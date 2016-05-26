from keras.layers import Input, Dense
from keras.models import model_from_json
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from Queue import Queue
import numpy as np

class NFQ:
  
  # in size = number of encoded features
  # out_size = number of possible actions
  def __init__(self, in_size, out_size, learning_rate = 0.975, model_path = None, weights_path = None):
    assert in_size > 1
    assert type(in_size) is int
    assert out_size > 1
    assert type(out_size) is int

    self.in_size = in_size
    self.out_size = out_size
    self.gamma = learning_rate

    if model is None:
      self.model = Sequential()
      self.model.add(Dense(64, input_dim=in_size, init='uniform'))
      self.model.add(Activation('tanh'))
      self.model.add(Dropout(0.2))
      self.model.add(Dense(64, init='uniform'))
      self.model.add(Activation('tanh'))
      self.model.add(Dropout(0.2))
      self.model.add(Dense(out_size, init='uniform'))
      self.model.add(Activation('softmax'))

      self.model.compile(loss='mse', # maybe binary_crossentrpy?
        optimizer='rmsprop',
        metrics=['accuracy'])
    else:
      assert weights_path is not None
      self.model = model_from_json(open(model_path).read())
      self.model.load_weights(weights_path)

    self.transitions = Queue(1000)

  def train(self):
    np_data = np.array(list(this.transitions.queue))
    # trim the input data of neural net because it also contains unwanted state' and reward information => need it only for target Q
    in_data = np.delete(np_data, np.s_[self.in_size::], 1)
    out_data = self.get_training_data(np_data)
    # stop_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    self.model.fit(in_data, out_data,
          nb_epoch=100,
          batch_size=16)
    # callbacks=[stop_cb]

  def predict_action(self, state):
    q = self.model.predict(state)
    return np.argmax(q)

  def get_training_data(self, data):
    out_data = list()
    for row in data:
      reward = row[-1]
      selected_action = row[self.in_size]
      next_state = row[self.in_size+1:-1]
      predicted_Q = self.model.predict(next_state)
      maxQ = np.max(predicted_Q)
 #     id_maxQ = np.argmax(predicted_Q)
      out = np.zeros((self.out_size,))
      out[selected_action] = reward + gamma*maxQ
      out_data.append(out)

    return np.array(out_data)

  # form [st, a, st+1, r]
  def add_transition(self, transition):
    # transition must be a numpy array
    assert type(transition) is np.ndarray
    if self.transitions.full():
      self.transitions.get()
    
    self.transitions.put(transition)

  def save_net(self, model_path = None, weights_path = None):
    if model_path is None:
      model_path = 'model.json'

    if weights_path is None:
      weights_path = 'weights.h5'

    try:
      self.model.save_weights(weights_path)
      open(model_path, 'w').write(self.model.to_json())
    except Exception as e:
      print('Saving a model failed')
      print(type(e))
      print(e)