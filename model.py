from keras.models          import Model
from keras.layers          import Input, Dense, GRU, LSTM, Dropout
# from keras.layers.merge    import Add, Multiply
from keras.layers          import add, multiply

from keras.layers.noise    import GaussianNoise as GN



class VAE():
    def __init__(self):
        self.input_dim = 5
        self.output_dim = 5
        self.latent_dim = 2

        
    def build_encoder(self):
        encoder_inputs = Input(shape=(None, self.input_dim))
        encoder_dense_outputs = Dense(self.input_dim, activation='relu')(encoder_inputs)
        myu = Dense(self.latent_dim, activation='sigmoid')(encoder_dense_outputs)
        sigma = Dense(self.latent_dim, activation='sigmoid')(encoder_dense_outputs)
        return Model(encoder_inputs, [myu, sigma]) 


    def build_decoder(self):
        # decoder_inputs = Input(shape=(None, self.input_dim))
        decoder_inputs = Input(shape=(None, self.latent_dim))
        decoder_dense_outputs = Dense(self.input_dim, activation='relu')(decoder_inputs)
        decoder_outputs = Dense(self.output_dim, activation='sigmoid')(decoder_dense_outputs)
        return Model(decoder_inputs, decoder_outputs) 


    def build_autoencoder(self, encoder, decoder):
        # encoder
        ei, ed, myu_l, sigma_l = encoder.layers
        encoder_inputs = Input(shape=(None, self.input_dim))
        encoder_dense_outputs = ed(encoder_inputs)
        myu = myu_l(encoder_dense_outputs)
        sigma = sigma_l(encoder_dense_outputs)

        # apply gaussiann
        g_inputs = Input(shape=(None, self.latent_dim))
        ipsilon = GN(stddev=1)(g_inputs)
        encoder_outputs = add([myu, multiply([ipsilon, sigma])])

        # decoder
        di, dd, do = decoder.layers
        decoder_dense_outputs = dd(encoder_outputs)
        decoder_outputs = do(decoder_dense_outputs)

        return Model([encoder_inputs, g_inputs], decoder_outputs)


    def save_model_fig(self, model, fname):
        import pydot
        from keras.utils import plot_model
        plot_model(model, to_file=fname)


def main():
    vae = VAE()
    en = vae.build_encoder()
    de = vae.build_decoder()
    ae = vae.build_autoencoder(en, de)
    ae.summary()

    # vae.save_model_fig(ae, "./ae.png")    
    # for value in ae.layers:
    #     print(value)


if __name__ == "__main__":
    main()
