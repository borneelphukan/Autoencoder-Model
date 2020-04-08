from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras
import numpy as np

class Autoencoder:
	@staticmethod
	def build(width, height, depth, filters=(32, 64), latentDim=16):

		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)
		x = inputs

		for f in filters:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
            #x = Conv2D(f, (3, 3), strides = 2, padding = "same")(x)
            #x = LeakyReLU(alpha = 0.2)(x)
            #x = BatchNormalization(axis = -1)(x)
		volumeSize = keras.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		encoder = Model(inputs, latent, name="encoder")

		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
            #x = Conv2DTranspose(f, (3,3), strides = 2, padding="same")(x)
            #x = LeakyReLU(alpha = 0.2)(x)
            #x = BatchNormalization(axis = -1)(x)
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)
		decoder = Model(latentInputs, outputs, name="decoder")
        
        #Combining Encoder and Decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)), name = "autoencoder")

		return (encoder, decoder, autoencoder)