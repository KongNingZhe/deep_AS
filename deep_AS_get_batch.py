# -*- coding: UTF-8 -*-
import linecache
import numpy as np
import tensorflow as tf

def read_data(genefilePath):
	read = open(genefilePath,"r")
	data = read.readlines()
	read.close()
	return data
#得到x的onehot batch 和y 的 batch 
#输入参数是这个data的list batch长度和起始hangshu
def get_batch(lines,batch_size,start_line):
	#decodedic = {'A':0,'T':1,'C':2,'G':3,'N':4}
	decodedic = {'A':[0,0,0,1,0],'T':[0,0,1,0,0],'C':[0,1,0,0,0],'G':[1,0,0,0,0],'N':[0,0,0,0,1]}
	x_batch = []
	y_batch = []
	for n in range(start_line,start_line+batch_size):
		line = lines[n]
		x = line.split()[0]
		y = line.split()[1]
		genedata = [decodedic[j] for j in x]
		x_batch.append(genedata)
		y_batch.append(y)
		#data_xs.append(x_one_hot)
	return (np.array(x_batch),np.array(y_batch))

##################################################################
def get_targets_one_hot_batch(genefilePath,file_lines,batch_size,site_num=1):
	decodedic = {'0':0,'1':1}
	labeldata = []
	data_ys = []
	for n in range(1,file_lines+1):
		line = linecache.getline(genefilePath,n)
		line = line.split()[1]
		decode = [decodedic[y] for y in line]
		labeldata.append(decode)
	data_y = np.zeros([file_lines], dtype=np.int32)
	y = np.zeros([batch_size,site_num],file_lines, dtype=np.int32)
	steps_length = file_lines//batch_size
	for steps in range(file_lines):
		data_y[steps:] = labeldata[steps]
	for i in range(steps_length):
		y = data_y[i * batch_size:(i + 1)*batch_size ,:]
		#y = tf.one_hot(y, 2)
		#data_ys.append(y)
		yield y
	return targets


def gen_epochs(n,genefilePath,batch_size):
	epoch = []
	for i in range(n):
		data = read_data(genefilePath)
		time = len(data)//batch_size
		notyield=[]
		for j in range(time):
			notyield.append(get_batch(data,batch_size,j*batch_size))
		epoch.append(notyield)
	return epoch
