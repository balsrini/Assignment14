# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 01:05:57 2019

@author: Balaji
"""
import os
import re
import shutil
import pickle
import tarfile
import tensorflow as tf

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10

def decode(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'image': tf.FixedLenFeature([],tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)                    
                    }
            )   
    
    image = tf.decode_raw(features['image'],tf.uint8)    
    image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
    print(image.shape)
    image = tf.reshape(image,[IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
    image = tf.cast(tf.transpose(image,[1,2,0]),tf.float32)
    
    #label = tf.decode_raw(features['label'],tf.int32)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label,10)
    return image,label

def preprocess_image(image, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image

def parse_record(file_names='C:/Balaji/ML/Assignment14/cifar-10/train.tfrecords',mode = tf.estimator.ModeKeys.EVAL,batch_size=128,num_epoch=1):
    dataset = tf.data.TFRecordDataset(filenames=file_names)
    
    dataset = dataset.map(decode)
    dataset = dataset.map(
      lambda image, label: (preprocess_image(image), label))
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.shuffle(1000 + 3*batch_size)
    dataset = dataset.batch(batch_size)


    images, labels = dataset.make_one_shot_iterator().get_next()

    features = {'images': images}
    return features, labels
    
train_data_files = ['C:/Balaji/ML/Assignment14/cifar-10/train.tfrecords']
(features,label) = parse_record()
    
    
        
    