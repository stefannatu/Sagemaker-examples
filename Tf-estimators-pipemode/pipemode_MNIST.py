#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:30:50 2019

@author: stenatu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:31:44 2019

@author: stenatu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:19:09 2019

@author: stenatu

In this code I show how to modify PipeMode dataset to work with script mode,
since the py2 version of PipeMode (and all the notebook examples in the documentation)
will be deprecated shortly.

"""
import tensorflow as tf
from sagemaker_tensorflow import PipeModeDataset
import os
import argparse
import logging

HEIGHT = 28
WIDTH = 28
DEPTH = 1

NUM_PARALLEL_BATCHES = 10
INPUT_TENSOR_NAME = 'inputs_input'
NCLASSES = 10# or num classes if problem is not binary classification

# Optionally add configs for estimator function here
#CHECKPOINT_STEPS = None
# TO BE ADDED


logging.getLogger().setLevel(logging.INFO)

def serving_input_fn():
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, DEPTH*HEIGHT*WIDTH])}
    return tf.estimator.export.ServingInputReceiver(features = inputs, receiver_tensors = inputs)
    
# Create training and eval input functions. 
def train_input_fn(params):
    """Returns input function that would feed the model during training"""
    return read_dataset('train', args.batch_size)

def eval_input_fn(params):
    """Returns input function that would feed the model during evaluation"""
    return read_dataset('eval', args.batch_size) # set epochs to 1

# Create a read dataset function which returns _input_fn. This is crucial 
# for canned estimators to make sure you don't get a "Not Callable" error.

def read_dataset(channel, batch_size):
    def _input_fn():
        def _read_and_decode(record):
            features = tf.parse_single_example(
                    record,
                    features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                     })

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape([HEIGHT*WIDTH*DEPTH])
            image = tf.cast(image, tf.float32) * (1. / 255)
            label = tf.cast(features['label'], tf.int32)

            return {INPUT_TENSOR_NAME: image}, label

    
        ds = PipeModeDataset(channel, record_format='TFRecord')
        ds = ds.repeat()
        ds = ds.prefetch(batch_size)
        ds = ds.map(_read_and_decode, num_parallel_calls = NUM_PARALLEL_BATCHES)
        
        if channel == 'train':
            ds= ds.shuffle(buffer_size = batch_size)

        ds = ds.batch(batch_size, drop_remainder = True)
        ds = ds.make_one_shot_iterator().get_next()
    
        return ds
    return _input_fn


if __name__=='__main__':
    parser = argparse.ArgumentParser()
#    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--model-dir', type = str, default = os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10000) # max_steps parameter for training
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_VAL'))


    args, _ = parser.parse_known_args()
#    
    logging.info('getting data')
#    
    train_spec = tf.estimator.TrainSpec(train_input_fn(params = None), max_steps = args.steps)
#    val_dataset = eval_input_fn(params = None)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn(params = None))
#    
#    
    logging.info("Building Tensorflow Estimator model")

#  specify features columns here. In this case we only have one, or else specify as a list. 
    
    column = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[HEIGHT*WIDTH*DEPTH])]
    
# Build a simple linear classifier -- just change the code to whatever canned estimator you want to use.    
    estimator = tf.estimator.LinearClassifier(feature_columns=column, 
                                         model_dir = args.model_dir,
                                         n_classes=NCLASSES,
                                         config=None)
    
    
    logging.info("Train and Evaluate model using TrainSpec, EvalSpec")
    
    tf.estimator.train_and_evaluate(estimator,train_spec, eval_spec)
 
    
    feature_spec = tf.feature_column.make_parse_example_spec(column)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(export_dir_base = args.model_dir, 
                                assets_extra = None,
                                serving_input_receiver_fn = export_input_fn, 
                                as_text=False)
    
    