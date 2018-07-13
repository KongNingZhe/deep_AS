import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf

import deep_AS_main_model
#import deep_AS_TrainModel
#import deep_AS_
import deep_AS_config
#import deep_AS_model
#import deep_AS_io
import deep_AS_train
#import deep_AS_test

def main(_):
    if deep_AS_config.FLAGS.model == "train":
        deep_AS_main_model.train()
    elif deep_AS_config.FLAGS.model == "test":
        deep_AS_main_model.test()
    elif deep_AS_config.FLAGS.model == "RNNtrain":
        deep_AS_main_model.LSTMtrain()
if __name__ == "__main__":
    tf.app.run()
