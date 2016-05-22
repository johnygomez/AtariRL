from Autoencoder import Autoencoder
from keras.layers import Input, Dense
from keras.models import Model_from_json
from keras.models import Model


class Encoder(Autoencoder):
	def __init__(self, pretrained_weights = True):
		self.Autoencoder = Model_from_json(open('PreTrained_model/model.json').read())
		if pretrained_weights:
			self.Autoencoder.load_weights('PreTrained_model/NN_weights.h5')
		self.Encoder = Model(input = self.Autoencoder.layers[0], output = self.Autoencoder.layers[1])
		self.Decoder = Model(input = self.Autoencoder.layers[1], output = self.Autoencoder.layers[2])

	def encode(self, img):
		return self.Encoder.predict(img)

	def decode(self, img):
		return self.Decoder.predict(img)

	
