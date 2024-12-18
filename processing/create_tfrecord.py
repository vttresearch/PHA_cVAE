import sys
import os
import numpy as np
import tensorflow as tf
import collections
import pickle
import uuid
import random
import pandas as pd
from Bio import SeqIO
import trRosetta_results

def extract_features(ref_row, path, file, seq):

    condition = trRosetta_results.get_condition(ref_row["class"])
    condition = np.array(condition.reshape(1,1,85), dtype=np.bool)

    npz_file= file.replace(".fasta",".npz")
    print(npz_file)
    trrosetta =trRosetta_results.create_distance_matrix(path+"rosetta/"+npz_file)
    trrosetta = np.array(trrosetta.reshape(1, 700, 75), dtype=np.float16)

    sec_file = file.replace(".fasta", ".horiz")
    sec_struct = trRosetta_results.process_secondary_structure_files(path+"secondary/"+sec_file)
    sec_struct = np.array(sec_struct.reshape(1, 700, 3), dtype=np.bool)

    chem_features = trRosetta_results.get_chemical_features(seq)
    chem_features = np.array(chem_features.reshape(1, 700, 7), dtype=np.float16)

    sequence = np.array(trRosetta_results.get_binary_sequence(seq), dtype=np.float16)


    return trrosetta, chem_features, sec_struct, condition, sequence


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createTFRecord(trrosetta, chem_feature, sec_struct, condition, sequence):
    """
    This function generates the record file with the features as content
    """

    data = tf.train.Example(features=tf.train.Features(
        feature={
                 'trrosetta': _bytes_feature(tf.io.serialize_tensor(trrosetta)),
                 'chemical features': _bytes_feature(tf.io.serialize_tensor(chem_feature)),
                 'secondary structure': _bytes_feature(tf.io.serialize_tensor(sec_struct)),
                 'condition': _bytes_feature(tf.io.serialize_tensor(condition)),
                 'sequence': _bytes_feature(tf.io.serialize_tensor(sequence))
                 }
        ))

    return data.SerializeToString()

def writeTFRecords(filename, dataset):

    with tf.io.TFRecordWriter(filename, 'GZIP') as writer:
        for contact_map, chem_feature, sec_struct, condition, sequence in dataset:
            writer.write(createTFRecord(contact_map, chem_feature, sec_struct, condition, sequence))


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
    sec_struct = tf.io.parse_tensor(samples['secondary structure'], out_type=tf.bool)
    condition = tf.io.parse_tensor(samples['condition'], out_type=tf.float16)
    sequence = tf.io.parse_tensor(samples['sequence'], out_type=tf.float16)

    return trrosetta, chem_features, sec_struct, condition, sequence





def main(input_dir, output_dir):
    reference = pd.read_csv("/scratch/project_2005440/seq_for_training/sequences_for_training_and_testing.csv")
    files = os.listdir(input_dir)
    for j in range(1000):
        trrosetta = np.array([])
        chem_features = np.array([])
        sec_struct = np.array([])
        condition = np.array([])
        sequence = np.array([])
        outfile = str(uuid.uuid4()) + ".tfrecord"


        file_choosen = random.choice(files)
        print(file_choosen)
        try:
            seq = ""
            seq_id = ""
            for record in SeqIO.parse(input_dir+file_choosen, "fasta"):
                seq = str(record.seq)
                seq_id = record.id
            if len(seq)< 10 or len(seq) > 700:
                print("Id:",seq_id)
                continue
            ref_row = reference[reference["ID"]==seq_id]

            rtr, aaf, ss, c, seq = extract_features(ref_row.iloc[0],input_dir, file_choosen, seq)
            if trrosetta.shape[0] == 0:
                trrosetta = rtr
                chem_features = aaf
                sec_struct = ss
                condition = c
                sequence = seq
            else:
                trrosetta = np.concatenate((trrosetta, rtr), axis=0)
                chem_features = np.concatenate((chem_features, aaf), axis=0)
                sec_struct = np.concatenate((sec_struct, ss), axis=0)
                condition = np.concatenate((condition, c), axis=0)
                sequence = np.concatenate((sequence, seq), axis=0)
        except:
            print("failed",file_choosen)
    print(trrosetta.shape)
    print(sec_struct.shape)
    print(chem_features.shape)
    print(condition.shape)
    print(sequence.shape)
    print(outfile)

    ds = tf.data.Dataset.from_tensor_slices((trrosetta, chem_features, sec_struct, condition, sequence))
    #ds = ds.shuffle(1000, reshuffle_each_iteration=False)
    writeTFRecords(output_dir + os.sep + outfile, ds)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage npy2tfrecord input_dir output_dir')
    main(sys.argv[1], sys.argv[2])