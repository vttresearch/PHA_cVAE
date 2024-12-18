import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras as k
from aux import *             # load all functions from auxiliary file to remove a bit of clutter
from models import *




def main():
    inputdatadir="input_files/training_5_new"
    testdatadir = "input_files/testing_5_new"


    d19e = Encoder()
    d19e.model = d19e.Build()
    d19e.model.summary()
    d19d = Decoder()
    d19d.model = d19d.Build()
    d19d.model.summary()

    encoder = tf.keras.models.load_model("encoder_lstm_transformer_heads_18_input_5_new_new_loss_v6")
    decoder = tf.keras.models.load_model("decoder_lstm_transformer_heads_18_input_5_new_new_loss_v6")

    x = layers.Input(shape=(700, 75))
    x2 = layers.Input(shape=(700, 7))
    x3 = layers.Input(shape=(700, 3))
    c = layers.Input(shape=(85))
    mean, log_var = encoder([x, c,x2,x3])


    z = layers.Lambda(sampling, output_shape=(latent_dim,))([mean, log_var])
    y2 = decoder([z, c])
    cvae = k.Model(inputs=[x, c,x2,x3], outputs=[y2], name='cvae')

    cvae.add_loss(KL_loss(mean,log_var))
    cvae.add_metric(KL_loss(mean,log_var), name='kl_loss', aggregation='mean')
    cvae.compile(optimizer=k.optimizers.Adam(learning_rate=0.001, decay=1e-6),loss=tf.nn.softmax_cross_entropy_with_logits, metrics="categorical_accuracy")
 


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


                cvae.fit(x=x,y=y, batch_size=batch_size, shuffle=False, epochs=1, verbose=1, validation_data=(x_test,y_test))

            except:
                raise
                
        saving_number = saving_number + 1
        decoder.save("decoder_lstm_transformer_heads_18_input_5_new_new_loss_v7")
        encoder.save("encoder_lstm_transformer_heads_18_input_5_new_new_loss_v7")
        print("Saved model-" + str(saving_number) + " to disk")
                


if __name__ == '__main__':

    main()
