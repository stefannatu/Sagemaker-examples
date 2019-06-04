#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:23:39 2019

@author: stenatu

This file converts the MNIST data into TfRecordIO format.
"""

import os
import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Have the option to shard a dataset if it doesn't fit in memory when converting to .tfrecord

def process_data(dataset, name, directory, num_shards=1):
    num_examples, rows, cols, depth = dataset.images.shape
    dataset = list(zip(dataset.images, dataset.labels))
    
    def _convert_to(data_set, name, directory):
        """Converts a dataset to tfrecords."""
        filename = os.path.join(directory, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index, (image, label) in enumerate(data_set):
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(label)),
                    'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())
        writer.close()
    
    if num_shards ==1:
        _convert_to(dataset, name, directory)
    else:
        sharded_dataset = np.array_split(dataset, num_shards)
        for shard, dataset in enumerate(sharded_dataset):
            _convert_to(dataset, f'{name}-{shard+1}', directory) 
     