import tensorflow as tf
import os

tf.config.run_functions_eagerly(True)

latent_dim = 20
batch_size = 12 

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


def KL_loss(z_mean, z_log_var):
    kl_loss = 1 * tf.reduce_mean(
        - 0.5 * 1 / latent_dim * tf.math.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=-1))

    return kl_loss*1


def ls_loss(x):
    out=tf.math.exp(-(1/0.0001)*tf.math.abs(x))
    return tf.reduce_mean(out)
		
		
		
def process_data(tfrecord):
    """
    This function takes an input from the dataset and parses it and augments it
    with the protein sequence
    """
    print(type(tfrecord))

    cm, cf, ss, ap, hc, co, sq = parse_data(tfrecord)

    return ([cm, co, cf, ss, ap, hc], sq)


def parse_data(tfrecord):
    """
    This function decodes the information from the dataset
    """
    features={
            'trrosetta': tf.io.FixedLenFeature((), tf.string),
            'chemical features': tf.io.FixedLenFeature((), tf.string),
            'secondary structure': tf.io.FixedLenFeature((), tf.string),
            'condition': tf.io.FixedLenFeature((), tf.string),
            'sequence': tf.io.FixedLenFeature((), tf.string),
            }
    samples = tf.io.parse_single_example(tfrecord, features)

    trrosetta = tf.io.parse_tensor(samples['trrosetta'], out_type=tf.float16)
    chem_features = tf.io.parse_tensor(samples['chemical features'], out_type=tf.float16)
    sec_struct = tf.io.parse_tensor(samples['secondary structure'], out_type=bool)
    condition = tf.io.parse_tensor(samples['condition'], out_type=bool)
    sequence = tf.io.parse_tensor(samples['sequence'], out_type=tf.float16)

    return trrosetta, chem_features, sec_struct, condition, sequence



def create_data_pipeline(inputdata_dir, batch_size, prefetch=2):
    """
    This function takes all input files located in direcotry inputdata_dir and
    generates a datapipe, the input files are expected to be of TFDataRecord
    format
    """
    input_files_names = tf.data.Dataset.list_files(inputdata_dir + os.sep + '*.tfrecord')
    input_files_names =  input_files_names.with_options(options)   

    ds = tf.data.TFRecordDataset(input_files_names, compression_type='GZIP', num_parallel_reads=4)
    ds = ds.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

