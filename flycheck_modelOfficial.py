from keras.models          import Model
from keras.layers          import Input, Dense, GRU, LSTM, Dropout, add, multiply
from keras.layers.core     import Lambda
from keras.layers.noise    import GaussianNoise as GN
from keras import backend as K


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


epsilon_std = 1.0 # if epsilon is about 0, VAE is closing to AutoEncoder

class Const():
    nb_epoch=30
    batch_size=100


class VAE():
    def __init__(self):
        self.input_dim = 748
        self.latent_dim = 100
        self.z_dim = 2


    def build_encoder(self):
        x = Input(shape=(self.input_dim, ))
        hidden = Dense(500, activation='relu')(x)
        z_mean = Dense(self.z_dim, activation='linear')(hidden)
        z_sigma = Dense(self.z_dim, activation='linear')(hidden)
        return Model(x, [z_mean, z_sigma])


    def build_decoder(self):
        z_mean = Input(shape=(self.z_dim, ))
        z_sigma = Input(shape=(self.z_dim, ))
        z = Lambda(self.sampling, output_shape=(self.z_dim,))([z_mean, z_sigma])
        h_decoded = Dense(500, activation='relu')(z)
        x_decoded_mean = Dense(784, activation='sigmoid')(h_decoded)
        return Model([z_mean, z_sigma], x_decoded_mean)


    def build_vae(self, encoder, decoder):
        _, ed, ed_m, ed_s = encoder.layers
        _, _, dl, dd1, dd2 = decoder.layers

        x = Input(shape=(self.input_dim, ))
        hidden = ed(x)
        z_mean = ed_m(hidden)
        z_sigma = ed_s(hidden)

        z = dl([z_mean, z_sigma])
        h_decoded = dd1(z)
        x_decoded_mean = dd2(h_decoded)
        return Model(x, x_decoded_mean)


    def build_gen(self, decoder):
        _, _, dl, dd1, dd2 = decoder.layers
        decoder_input = Input(shape=(self.latent_dim,))
        h_decoded = dd1(decoder_input)
        x_decoded_mean = dd2(h_decoded)
        return Model(decoder_input, x_decoded_mean)


    def sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(self.z_dim,), mean=0., stddev=epsilon_std)
        return z_mean + z_sigma * epsilon


    def train_model(self, model, X_train, X_test):
        hist_vae = model.fit(X_train, X_train,
                           nb_epoch=Const.nb_epoch,
                           batch_size=Const.batch_size,
                           shuffle = True,
                           validation_data=(X_test, X_test))


    def binary_crossentropy(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


    def vae_loss(self, x, x_decoded_mean):
        _, _, z_mean, z_sigma, _ ,_, x_decoded_mean = vae.layers
        x = Input(shape=(self.input_dim, ))
        reconst_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean),axis=-1)
        latent_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1))
        return reconst_loss + latent_loss


    def model_compile(self, model):
        model.compile(optimizer='rmsprop', loss=self.vae_loss(model))


def main():
    vae = VAE()
    encoder = vae.build_encoder()
    decoder = vae.build_decoder()    
    vae_model = vae.build_vae(encoder, decoder)
    vae.model_compile(vae_model)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    vae.train_model(vae_model, x_train, x_test)

    x_test_encoded = encoder.predict(x_test, batch_size=Const.batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
   main()
