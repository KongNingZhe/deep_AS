import deep_AS_get_batch
import deep_AS_train
import deep_AS_model 
import tensorflow as tf
import deep_AS_config
import matplotlib.pyplot as plt
#from tensorflow.python.framework import ops


def get_logits(x):
	#cnn_input = tf.reshape(x,[deep_AS_config.batch_size, deep_AS_config.num_units, deep_AS_config.vocab_size,1])
	
    softmax_weight1 = tf.get_variable(
                                    name="dense8_weights", 
                                    shape=[deep_AS_config.FLAGS.num_units, deep_AS_config.FLAGS.label_class],
                                    initializer=tf.uniform_unit_scaling_initializer(1.43))
    softmax_bias1 = tf.get_variable(
                                name="dense8_biases", 
                                shape=[deep_AS_config.FLAGS.label_class],
                                initializer=tf.constant_initializer(0.1))
    #logit = tf.nn.relu(tf.matmul(out, dense8_weight) + dense8_bias)
    logit= tf.softmax(tf.matmul(x, softmax_weight1) + softmax_bias1)
    return logit

def loss_per_batch(x,y):
	#for logit, target in zip(logits, targets):
	logits = get_logits(x)
	targets = y
	losses = [nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=target) for logit, target in zip(logits, targets)]
	total_loss = tf.reduce_mean(losses)
	return total_loss

def accuracy(x,y):
	logit = get_logits(x)
	predictions=tf.nn.softmax(logit)
	y_true = tf.one_hot(y, 2)##[batch_size,2]
	y_true = tf.cast(y_true,tf.float32)
	correct_pred = tf.equal(tf.argmax(predictions, 1),tf.argmax(y_true, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	return accuracy

def train(input):

	return output
