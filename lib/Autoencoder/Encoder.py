from Autoencoder import Autoencoder
from keras.layers import Input, Dense
from keras.models import model_from_json
from keras.models import Model
from keras.utils.layer_utils import layer_from_config


class Encoder(Autoencoder):
	def __init__(self, pre_trained_model, path_to_model, path_to_weights):		
		self.autoencoder = model_from_json(open(path_to_model).read())
		self.autoencoder.load_weights(path_to_weights)
		self.autoencoder.summary()
		self.input_img = Input(shape=(1600,))
		self.encoded = Dense(self.autoencoder.layers[1].output_dim, activation=self.autoencoder.layers[1].activation, weights = self.autoencoder.layers[1].get_weights())(self.input_img)
		self.encoder = Model(input = self.input_img, output = self.encoded)
		self.encoder.summary()

	def encode(self, img):
		p = self.encoder.predict(img)
		print p.shape
		return p

	def draw(self, img):
		predicted_img = self.autoencoder.predict(img)
		import matplotlib.pyplot as plt
		plt.figure(figsize=(1,1), dpi=40)
		plt.imshow(predicted_img.reshape(40, 40))
		plt.show()


	def decode(self, img):
		return self.decoder.predict(img)

	def fit(self, data, n_epochs = 1000, val_split = 0.1, batch_s = 512):
		if use_pre_trained_model:
			print 'Fit is not necessary!'
			return
		autoencoder.fit(data, data,
                nb_epoch=n_epochs,
                batch_size=batch_s,
                shuffle=True,
                validation_split = val_split
                )

	
