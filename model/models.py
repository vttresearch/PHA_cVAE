import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
                                            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
                                            Reshape, GlobalAveragePooling2D, Layer, Multiply, Dense)
from tensorflow.python.keras.models import Sequential, Model
#from tensorflow import keras as k
#from keras import backend as K


latent_dim = 20


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    '''z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon'''

    mean, log_var = args
    eps = tf.random.normal(shape=(tf.shape(mean)[0], latent_dim), mean=0., stddev=1.0)
    return mean + tf.exp(log_var / 2.) * eps


class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    # return conv + bn + leaky_relu model
    def __call__(self, net, name="", training=None):
        net = Conv2D(self.filters, self.kernelSize, strides=self.strides, padding='same', name=name)(net)
        net = BatchNormalization()(net, training=training)
        net = LeakyReLU()(net)
        return net

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        norminput = self.layernorm1(inputs)
        attn_output = self.att(norminput, norminput)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Darknet19Encoder():

    def __init__(self):

        self.latentSize = 20

    def Build(self):
        

        num_heads = 18  # Number of attention heads
        ff_dim = 64  # Hidden layer size in feed forward network inside transformer        
        
        # create the input layer for feeding the netowrk
        x = Input(shape=(700,75))
        c = layers.Input(shape=(85))
        print(c.shape)
        #c = layers.Flatten()(c)
        print(c.shape)

        mask1 = layers.Masking(mask_value=0., input_shape =(700,75))(x)
        embed_dim = 75 # Embedding size for each token
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        t1 = transformer_block(mask1)
        d1 = Dropout(0.1)(t1)        
        lstm1 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d1)
        #print(lstm1.shape)

        x2 = Input(shape=(700, 7))
        mask2 = layers.Masking(mask_value=0., input_shape =(700,7))(x2)
        embed_dim = 7 # Embedding size for each token
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        t2 = transformer_block(mask2)
        d2 = Dropout(0.1)(t2)
        lstm2 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d2)        

        x3 = Input(shape=(700, 3))
        mask3 = layers.Masking(mask_value=0., input_shape =(700,3))(x3)
        embed_dim = 3 # Embedding size for each token
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        t3 = transformer_block(mask3)
        d3 = Dropout(0.1)(t3)        
        lstm3 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d3)

		
        netconcatenated = layers.concatenate([lstm1, lstm2, lstm3, c], axis=1)
        print(netconcatenated.shape)
        #net = layers.concatenate([netconcatenated,c], axis=1)
        
        net = tf.keras.layers.Dense(300)(netconcatenated)
        print("here")
        net = tf.keras.layers.Dense(100)(net)
        # net = tf.keras.layers.Dense(85)(net)
        # net = tf.keras.layers.Dense(30)(net)
        print("here")
        #net = tf.keras.layers.Dense(20)(net)
        #net = tf.keras.layers.Dense(10)(net)
        #net = tf.keras.layers.Dense(8)(net)
        #net = tf.keras.layers.Dense(6)(net)
        #net = tf.keras.layers.Dense(4)(net)
        #net = tf.keras.layers.Dense(3)(net)
        #net = tf.keras.layers.Dense(2)(net)
        #net = tf.keras.layers.Dense(1)(net)
        #y2 = tf.keras.activations.sigmoid(net)
        #print(y2.shape)
        
        mean = tf.keras.layers.Dense(self.latentSize)(net)
        print("yes")
        logvar = tf.keras.layers.Dense(self.latentSize)(net)
        print("YES")
        
        #mean = GlobalAveragePooling2D()(y2)
        #logvar = GlobalAveragePooling2D()(y2)

        # variational encoder output (distributions)
        #mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      #padding='same')(net)
        #mean = GlobalAveragePooling2D()(mean)
        #logvar = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                       # padding='same')(net)
        #logvar = GlobalAveragePooling2D()(logvar)

        # sample = SampleLayer(self.latentConstraints, self.beta)([mean, logvar], training=self.training)

        return Model(inputs=[x, c, x2, x3], outputs=[mean, logvar], name='encoder')


class Darknet19Decoder():
    def __init__(self):
        self.latentSize = 20

    def Build(self):
        z = layers.Input(shape=(self.latentSize,1))
        c = layers.Input(shape=(85,1))
        netcon = layers.concatenate([c, z], axis=1)
        print(netcon.shape)
        
        #net_3D = tf.keras.layers.RepeatVector(1)(netcon)
        #print(net_3D.shape)
        #net = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(net_3D)
        #net = layers.Conv2DTranspose(128, 1, activation="relu", strides=2, padding="same")(net)
        #net = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(net)
        #net = layers.Conv2DTranspose(16, 5, activation="relu", strides=2, padding="same")(net)
        
        net = layers.Bidirectional(tf.keras.layers.LSTM(256))(netcon)

        net2 = layers.Flatten()(net)
        net2 = layers.Dense(100, activation="relu")(net2)
        net2 = layers.Dense(700 * 21)(net2)

        net2 = Reshape((700, 21))(net2)
        net2 = layers.Lambda(lambda x: tf.cast(x, 'float32'), name='change_to_float')(net2)

        #net2 = layers.Softmax(axis=2)(net2)

        return Model([z, c], [net2], name='decoder')
