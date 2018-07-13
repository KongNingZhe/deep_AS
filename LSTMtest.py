import functools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import random
import time
import numpy as np


def lazy_property(function):
	attribute = '_' + function.__name__

	@property
	@functools.wraps(function)
	def wrapper(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)
	return wrapper


class VariableSequenceLabelling:

	def __init__(self, data, target, num_hidden=200, num_layers=3):
		self.data = data
		self.target = target
		self._num_hidden = num_hidden
		self._num_layers = num_layers
		self.prediction
		self.error
		self.optimize

	@lazy_property
	def length(self):
		used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length

	@lazy_property
	def prediction(self):
		# Recurrent network.
		output, _ = tf.nn.dynamic_rnn(
			tf.nn.rnn_cell.GRUCell(self._num_hidden),
			self.data,
			dtype=tf.float32,
			sequence_length=self.length,
		)
		# Softmax layer.
		max_length = int(self.data.get_shape()[1])
		num_classes = int(self.target.get_shape()[1])
		weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
		# Flatten to apply same weights to all time steps.
		output = tf.reshape(output, [-1, self._num_hidden])
		prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
		prediction = tf.reshape(prediction, [-1, max_length, num_classes])
		return prediction

	@lazy_property
	def cost(self):
		# Compute cross entropy for each frame.
		cross_entropy = self.target * tf.log(self.prediction)
		cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
		mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
		cross_entropy *= mask
		# Average over actual sequence lengths.
		cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
		cross_entropy /= tf.cast(self.length, tf.float32)
		return tf.reduce_mean(cross_entropy)

	@lazy_property
	def optimize(self):
		learning_rate = 0.0003
		optimizer = tf.train.AdamOptimizer(learning_rate)
		return optimizer.minimize(self.cost)

	@lazy_property
	def error(self):
		mistakes = tf.not_equal(
			tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
		mistakes = tf.cast(mistakes, tf.float32)
		mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
		mistakes *= mask
		# Average over actual sequence lengths.
		mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
		mistakes /= tf.cast(self.length, tf.float32)
		return tf.reduce_mean(mistakes)

	@staticmethod
	def _weight_and_bias(in_size, out_size):
		weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
		bias = tf.constant(0.1, shape=[out_size])
		return tf.Variable(weight), tf.Variable(bias)

def database(dir,max_len):
	f = open(dir, 'r')
	decodelabel = {"0": [1, 0],
				"1": [0, 1]}
	x_one_hot = []
	y_one_hot = []
	print("#####读取data")
	data = f.readlines()
	print("#####over data")
	print("训练集的大小有：  ", len(data))

	for line in data:
		x = line.split()[:-1]
		array = [[0]*64]*max_len
		if len(x) > max_len:
			continue
		for i in range(len(x)):
			array[i] = list(x[i])
		y = line.split()[-1]
		x_one_hot.append(array)
		y_one_hot.append(decodelabel[y])
		# xx = np.array(x_one_hot)
		# yy = np.array(y_one_hot)
	return x_one_hot,y_one_hot,len(y_one_hot)


def getbatch(x, y, lenth):
	x_batch = [] 
	y_batch = [] 
	batch_size = 4
	for i in range(batch_size):
		j = random.randint(0, lenth - 1)
		x_batch.append(x[j])
		y_batch.append(y[j])
	print (y_batch)
	return x_batch, y_batch

if __name__ == '__main__':
	#train, test = get_dataset()
	#_, length, image_size = train.data.shape
	#num_classes = train.target.shape[2]
	length = 100
	image_size = 64
	num_classes = 2
	dir = "../data64/t_mini"

	x,y,l = database(dir,length)


	data = tf.placeholder(tf.float32, [None, length, image_size])
	target = tf.placeholder(tf.float32, [None, num_classes])
	model = VariableSequenceLabelling(data, target)
	session = tf.Session()
	session.run(tf.global_variables_initializer())
	for epoch in range(1):
		for _ in range(1):
			#batch = train.sample(10)
			batch_x,batch_y = getbatch(x,y,l)
			session.run(model.optimize, {data: batch_x, target: batch_y})
		error = session.run(model.error, {data: test.data, target: test.target})
		print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
