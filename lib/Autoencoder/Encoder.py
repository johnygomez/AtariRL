from Autoencoder import Autoencoder
from keras.layers import Input, Dense
from keras.models import Model_from_json
from keras.models import Model


class Encoder(Autoencoder):
	def __init__(self, pre_trained_model = True, path_to_model = 'pre_trained_model/model.json', path_to_weights = 'pre_trained_model/NN_weights.h5', input_dim = 1600, encoding_dim = 25):		
		self.input_dim = input_dim
		self.out_dim = encoding_dim
		self.use_pre_trained_model = pre_trained_model
		if pre_trained_model:
			self.autoencoder = Model_from_json(open(path_to_model).read())
			self.autoencoder.load_weights(path_to_weights)
		else:
			input_img = Input(shape=(input_dim,))
			encoded = Dense(encoding_dim, activation='relu')(input_img)
			decoded = Dense(imput_dim, activation='sigmoid')(encoded)
			self.autoencoder = Model(input=input_img, output=decoded)
		self.autoencoder.summary()
		self.encoder = Model(input = self.autoencoder.layers[0], output = self.autoencoder.layers[1])
		self.decoder = Model(input = self.autoencoder.layers[1], output = self.autoencoder.layers[2])

	def encode(self, img):
		return self.encoder.predict(img)

	def decode(self, img):
		return self.decoder.predict(img)

	def fit(self, data, n_epochs = 1000, val_split = 0.1, batch_s = 512):
		if use_pre_trained_model:
			print 'Fit is not necessary!'
			return
		autoencoder.fit(data, data
                nb_epoch=n_epochs,
                batch_size=batch_s,
                shuffle=True,
                validation_split = val_split
                )

	
