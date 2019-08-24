# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:15:26 2019

@author: Balaji
"""
import os
import re
import shutil
import pickle
import tarfile
import tensorflow as tf

#CIFAR_FILENAME = 'cifar-10-python.tar.gz'
#CIFAR_DOWNLOAD_URL = 'http://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'C:/Balaji/ML/Assignment14/cifar-10'
CIFAR_NAME = 'cifar-10-batches-py'
CIFAR_LOCAL_COPY = 'C:/Balaji/ML/Assignment14/cifar-10-python.tar.gz'

def extractCifar10():    
    tarfile.open(CIFAR_LOCAL_COPY,'r:gz').extractall(CIFAR_LOCAL_FOLDER)

def get_file_name():
    file_names= {}
    file_names['train'] = ['data_batch_%d' %i for i in range(1,5)]
    file_names['validator'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names

def read_from_file(filename):
    with tf.gfile.Open(filename,'rb') as f:
        data_dict = pickle.load(f,encoding='latin1')
    return data_dict

def _bytes_feature(value):   
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_to_tfRecord(input_files,output_file):
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_from_file(input_file)
            data = data_dict['data']
            labels = data_dict['labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):                     
                example = tf.train.Example(features=tf.train.Features(
                                feature={
                                        'image':_bytes_feature(data[i].tobytes()),
                                        'label':_int64_feature(labels[i])
                                        }))
            record_writer.write(example.SerializeToString())
                        
                

extractCifar10()
file_names = get_file_name()
for mode,files in file_names.items():
    input_files = [os.path.join(CIFAR_LOCAL_FOLDER,CIFAR_NAME,f) for f in files]
    output_file = os.path.join(CIFAR_LOCAL_FOLDER,mode + '.tfrecords')
    try:
        os.remove(output_file)
    except OSError:
        pass
    convert_to_tfRecord(input_files,output_file)
    
