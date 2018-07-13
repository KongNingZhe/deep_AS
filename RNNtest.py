import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import random
import numpy as np


def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

def getbatch():
	a = [[[0,0,0],[0,0,0]],[[7,8,9],[1,2,3]]]
	return a
with tf.Session() as sess:
	data = tf.placeholder(tf.float32, [None,2,3])
	len = length(data)
	x = getbatch()
	t = sess.run(len, {data:x})
	print (t)
