import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
                                            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
                                            Reshape, GlobalAveragePooling2D, Layer, Multiply)
from tensorflow.python.keras.models import Model
from tensorflow import keras as k
import os, random
import pickle
import sys
import numpy as np
import collections
from aux import *             # load all functions from auxiliary file to remove a bit of clutter
from models import *

#strategy = tf.distribute.MirroredStrategy()


def main():
    inputdatadir="/scratch/project_2005440/input_files/training_5_new"
    testdatadir = "/scratch/project_2005440/input_files/testing_5_new"

    #with strategy.scope():

    d19e = Darknet19Encoder()
    print("HERE")
    d19e.model = d19e.Build()
    d19e.model.summary()
    d19d = Darknet19Decoder()
    d19d.model = d19d.Build()
    d19d.model.summary()


    # CVAE all together!
    #encoder =  d19e.model
    #decoder = d19d.model


    encoder = tf.keras.models.load_model("/scratch/project_2005440/tuulas_models/saved_models/encoder_lstm_transformer_heads_18_input_5_new_new_loss_v6")
    decoder = tf.keras.models.load_model("/scratch/project_2005440/tuulas_models/saved_models/decoder_lstm_transformer_heads_18_input_5_new_new_loss_v6")

    x = layers.Input(shape=(700, 75))
    x2 = layers.Input(shape=(700, 7))
    x3 = layers.Input(shape=(700, 3))
    c = layers.Input(shape=(85))
    mean, log_var = encoder([x, c,x2,x3])


    z = layers.Lambda(sampling, output_shape=(latent_dim,))([mean, log_var])
    print(z.shape)
    #z = layers.Reshape(latent_dim)(z)
    y2 = decoder([z, c])
    #checkpoint_path = "/scratch/project_2004076/workpad/scsandra/chem2bio/src/training_2/cp.ckpt"
    cvae = k.Model(inputs=[x, c,x2,x3], outputs=[y2], name='cvae')
    #cvae.load_weights(checkpoint_path)

    #cvae.add_loss(loss(x2,y2, mean,log_var,alpha=1, beta=0))
    #cvae.add_loss(ls_loss(mean))
    #cvae.add_metric(ls_loss(mean), name='latent_loss', aggregation='mean')

    cvae.add_loss(KL_loss(mean,log_var))
    cvae.add_metric(KL_loss(mean,log_var), name='kl_loss', aggregation='mean')
    cvae.compile(optimizer=k.optimizers.Adam(learning_rate=0.001, decay=1e-6),loss=tf.nn.softmax_cross_entropy_with_logits, metrics="categorical_accuracy")
    print(cvae.summary())



    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
    #                         save_weights_only=True,
    #                         verbose=1)


    ds = create_data_pipeline(inputdatadir, 980)
    test = create_data_pipeline(testdatadir, 980)
    saving_number = 0
    for e in range(1000):
        trrosetta, chem_features, sec_struct, condition, sequence = list(ds.take(1).as_numpy_iterator())[0]
        vtrrosetta, vchem_features, vsec_struct, vcondition, vsequence = list(test.take(1).as_numpy_iterator())[0]
        chem_feat_part = [chem_features[i:i + 98] for i in range(0, len(chem_features), 98)]
        trrosetta_part = [trrosetta[i:i + 98] for i in range(0, len(trrosetta), 98)]
        sec_struct_part = [sec_struct[i:i + 98] for i in range(0, len(sec_struct), 98)]
        condition_part = [condition[i:i + 98] for i in range(0, len(condition), 98)]
        sequence_part = [sequence[i:i + 98] for i in range(0, len(sequence), 98)]
        
        chem_feat_part_test = [vchem_features[i:i + 98] for i in range(0, len(vchem_features), 98)]
        trrosetta_part_test = [vtrrosetta[i:i + 98] for i in range(0, len(vtrrosetta), 98)]
        sec_struct_part_test = [vsec_struct[i:i + 98] for i in range(0, len(vsec_struct), 98)]
        condition_part_test = [vcondition[i:i + 98] for i in range(0, len(vcondition), 98)]
        sequence_part_test = [vsequence[i:i + 98] for i in range(0, len(vsequence), 98)]


        for i in range(10):
            print("epoch %d" % e)
            try:

                x = [trrosetta_part[i],
                     condition_part[i].reshape(len(condition_part[i]),85),
                     chem_feat_part[i],
                     sec_struct_part[i]]
                y = [sequence_part[i]]
                print(i)
                x_test = [trrosetta_part_test[i],
                     condition_part_test[i].reshape(len(condition_part_test[i]),85),
                     chem_feat_part_test[i],
                     sec_struct_part_test[i]]
                y_test = [sequence_part_test[i]]


                #vx = [vcontact_map,
                      #vcondition.reshape(len(vcondition),85),
                      #vchem_features,
                      #vsec_struct]
                #print(vsequence.shape)
                #print("batch_size")
                #print(batch_size)
                #print(trrosetta_part[i].shape)
                #print(condition_part[i].shape)
                #print(vcondition.shape)
                #print(y)
                cvae.fit(x=x,y=y, batch_size=batch_size, shuffle=False, epochs=1, verbose=1, validation_data=(x_test,y_test))

            except:
                raise
                print("something failed")
                
        saving_number = saving_number + 1
        decoder.save("/scratch/project_2005440/tuulas_models/saved_models/decoder_lstm_transformer_heads_18_input_5_new_new_loss_v7")
        encoder.save("/scratch/project_2005440/tuulas_models/saved_models/encoder_lstm_transformer_heads_18_input_5_new_new_loss_v7")
            # cvae.save("model-" + str(saving_number))
        print("Saved model-" + str(saving_number) + " to disk")
                


if __name__ == '__main__':

    main()
