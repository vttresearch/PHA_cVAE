import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.python.keras.layers import (Conv2D, BatchNormalization, LeakyReLU, Reshape, Layer,  Dense)
from tensorflow.python.keras.models import Sequential, Model


latent_dim = 20


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
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

class AttentionBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(AttentionBlock, self).__init__()
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

class Encoder():

    def __init__(self):

        self.latentSize = 20

    def Build(self):
        

        num_heads = 18  # Number of attention heads
        ff_dim = 64  # Hidden layer size in feed forward network inside transformer        
        
        # create the input layer for feeding the netowrk
        x = Input(shape=(700,75))
        c = layers.Input(shape=(85))

        mask1 = layers.Masking(mask_value=0., input_shape =(700,75))(x)
        embed_dim = 75 # Embedding size for each token
        transformer_block = AttentionBlock(embed_dim, num_heads, ff_dim)
        t1 = transformer_block(mask1)
        d1 = Dropout(0.1)(t1)        
        lstm1 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d1)

        x2 = Input(shape=(700, 7))
        mask2 = layers.Masking(mask_value=0., input_shape =(700,7))(x2)
        embed_dim = 7 # Embedding size for each token
        transformer_block = AttentionBlock(embed_dim, num_heads, ff_dim)
        t2 = transformer_block(mask2)
        d2 = Dropout(0.1)(t2)
        lstm2 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d2)        

        x3 = Input(shape=(700, 3))
        mask3 = layers.Masking(mask_value=0., input_shape =(700,3))(x3)
        embed_dim = 3 # Embedding size for each token
        transformer_block = AttentionBlock(embed_dim, num_heads, ff_dim)
        t3 = transformer_block(mask3)
        d3 = Dropout(0.1)(t3)        
        lstm3 = layers.Bidirectional(tf.keras.layers.LSTM(85, return_sequences=False, recurrent_dropout=0.1))(d3)

		
        netconcatenated = layers.concatenate([lstm1, lstm2, lstm3, c], axis=1)
        
        net = tf.keras.layers.Dense(300)(netconcatenated)
        net = tf.keras.layers.Dense(100)(net)
        
        mean = tf.keras.layers.Dense(self.latentSize)(net)
        logvar = tf.keras.layers.Dense(self.latentSize)(net)

        return Model(inputs=[x, c, x2, x3], outputs=[mean, logvar], name='encoder')


class Decoder():
    def __init__(self):
        self.latentSize = 20

    def Build(self):
        z = layers.Input(shape=(self.latentSize,1))
        c = layers.Input(shape=(85,1))
        netcon = layers.concatenate([c, z], axis=1)
        net = layers.Bidirectional(tf.keras.layers.LSTM(256))(netcon)

        net2 = layers.Flatten()(net)
        net2 = layers.Dense(100, activation="relu")(net2)
        net2 = layers.Dense(700 * 21)(net2)

        net2 = Reshape((700, 21))(net2)
        net2 = layers.Lambda(lambda x: tf.cast(x, 'float32'), name='change_to_float')(net2)
        return Model([z, c], [net2], name='decoder')
