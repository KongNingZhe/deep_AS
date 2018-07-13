import deep_AS_get_batch
import deep_AS_train
import tensorflow as tf
import deep_AS_config
import deep_AS_TrainModel

import LSTMclass as L

import os

import deep_AS_model

import random
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')#不展示图形界面

#global_step = tf.get_variable('global_step', dtype='int32',shape=[1],initializer=tf.constant_initializer(0), trainable=False)

def database(dir):
	f=open(dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	decodelabel = {"0":[1,0,0,0],
					"1":[0,1,0,0],
					"2":[0,0,1,0],
					"3":[0,0,0,1]}
	x_one_hot=[]
	y_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		arrayx=[]
		x = line.split()[0]
		x = x.upper()
		if x == 1 or x == 0:
			continue
		y = line.split()[1]
		for i in x:
			arrayx.append(decodedic[i])
		
		x_one_hot.append(arrayx)
		y_one_hot.append(decodelabel[y])
	return x_one_hot,y_one_hot,len(x_one_hot)

def database_snp(dir):
	f=open(dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	decodelabel = {"0":[1,0],
					"1":[0,1]
					}
	x_one_hot=[]
	y_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		line = line.strip()
		arrayx=[]
		x = line.split('\t')[0]
		x = x.upper()
		#print(x)
		if x == '1' or x == '0':
			continue
		y = line.split('\t')[1]
		for i in x:
			arrayx.append(decodedic[i])
		
		x_one_hot.append(arrayx)
		y_one_hot.append(decodelabel[y])
	print("去掉了部分杂质还有")
	print(len(x_one_hot))
	return x_one_hot,y_one_hot,len(x_one_hot)

def getbatch(x,y,lenth):
	x_batch=[]
	y_batch=[]
	for i in range(deep_AS_config.FLAGS.batch_size):
		j=random.randint(0,lenth-1)
		x_batch.append(x[j])
		y_batch.append(y[j])
	return x_batch,y_batch

def create_RNN_model(session):
	print("create_RNN_model()")
	
	rnn_model = L.VariableSequenceClassification()
	rnn_model.restore_model(session)
	return rnn_model

def create_model(session):

	print("create_model()")
	model = deep_AS_model.Model()
	model.restore_model(session)
	return model

def train_cycle(model,
				sess,
				epoch,
				checkpoint_path,
				x,
				y,
				l,
				log_file_handle,
				losses):
	# 调用model step
	# encoder_input >> step_loss output logists
	#print(step)
	current_step = 0
	training_loss = 0
	acc_total = 0
	
	start = time.clock()
	while True:
		input_x,input_y = getbatch(x,y,l)
		step_loss,acc = model.step(sess,input_x,input_y)
		losses.append(step_loss)
		global step
		step +=1
		training_loss +=step_loss
		acc_total += acc
		current_step+=1
		if current_step % deep_AS_config.FLAGS.steps_per_checkpoint == 0:

			training_loss = training_loss / deep_AS_config.FLAGS.steps_per_checkpoint
			acc_total = acc_total / deep_AS_config.FLAGS.steps_per_checkpoint

			print ('训练集:',"%d\t%d\t%.4f\t%.4f\n" % ( step,epoch,acc_total,training_loss),file=log_file_handle,end="")
			log_file_handle.flush()
			end = time.clock()
			cycletime  = end - start
			print("step\t",step)
			print("损失\t", training_loss)
			print("准确率\t", acc_total)
			print ("完成了一个cycle","time:",cycletime)
			break
	model.save_model(sess,checkpoint_path,step)
def train():
	print("train()")
	# TRAINING on train_set

	log_file = deep_AS_config.FLAGS.log_dir + "/log_file"
	print("Open log_file: ", log_file)
	log_file_handle = open(log_file, 'a')
	print("global step\tepoch\taccurancy\nloss\n",
			file=log_file_handle,
			end="")
	#make database
	print("make database")
	x,y,l = database_snp(deep_AS_config.FLAGS.train_dir)
	print("database done")

	# print("make test database")
	# x_test, y_test, l_test = database(deep_AS_config.FLAGS.test_dir)
	# print("test database done")

	from deep_AS_model import step

	with tf.Session() as sess:

		model = create_model(sess)
		global step
		checkpoint_path = os.path.join(deep_AS_config.FLAGS.model_dir, "data1.ckpt")
		writer = tf.summary.FileWriter('./graphs', sess.graph)
		epoch = 0
		losses = []
		while True:
			#规定训练轮数 大于训练轮数便停止
			epoch_last = epoch
			epoch = (step
			* deep_AS_config.FLAGS.batch_size
			//l)
			train_cycle(model,
						sess,
						epoch,
						checkpoint_path,
						x,
						y,
						l,
						log_file_handle,
						losses)

			if epoch >= deep_AS_config.FLAGS.epoch:
				print("EPOCH：")
				print(epoch)
				print("STOP TRAINING LOOP")
				break
			# iftest = epoch-epoch_last
			# if iftest:
			# 	print("".join(["="] * 80))  # section-separation line
			# 	print("have a test on test-data")
			# 	test_cycle(model,sess,x_test,y_test,l_test)

		#做图
		plt.plot(losses)
		plt.savefig("../G_bysj/log1/losses.pdf")

		#关闭句柄
		log_file_handle.close()
def test_cycle(model,sess,x_test,y_test,l):
	loss_t=0
	acc_t = 0
	input_x=[]
	input_y=[]
	number= 5 
	for n in range(number):
		for i in range(deep_AS_config.FLAGS.batch_size):
			j=random.randint(0,l-1)
			input_x.append(x_test[j])
			input_y.append(y_test[j])
		loss, acc = model.test_step(sess, input_x, input_y)
		acc_t += acc
		loss_t += loss
		input_x = []
		input_y = []
	print("测试——准确率\t", acc_t/number)
	print("测试——损失\t", loss_t/number)

def test():
	print("test() begin")
	log_file = deep_AS_config.FLAGS.log_dir + "/log_file_data1_test.tab"
	print("Open log_file: ", log_file)
	log_file_handle = open(log_file, 'w+')
	print("num \t loss\t accurancy\n",
			file=log_file_handle,
			end="")

	print("make test database")
	x,y,l = database(deep_AS_config.FLAGS.test_dir)
	print("database done")

	with tf.Session() as sess:
		print("Create model for test")
		print("change some para for test")
		deep_AS_config.FLAGS.batch_size = 1

		model = deep_AS_model.Model()
		model.build_model
		model.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(model.model_dir)
		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
			model.saver.restore(sess, ckpt.model_checkpoint_path)
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path),file=log_file_handle)
		else:
			print("error")
		
		acc_total = 0
		acc_100per = 0
		loss_total = 0
		loss_100per = 0
		input_x = []
		input_y = []
		for i in range(l):
			input_x.append(x[i])
			input_y.append(y[i])
			loss,acc = model.test_step(sess,input_x,input_y)
			acc_total += acc
			acc_100per += acc
			loss_total += loss
			loss_100per += loss
			input_x = []
			input_y = []
			if i %1000 == 0:
				print ("%d \t %.6f \t %.6f\n"
		 			% (
						i,
						loss_100per,
						acc_100per),
						file=log_file_handle,
						end="")
				log_file_handle.flush()
				print("个数\t",i)
				print("损失\t",loss_100per/1000)
				print("准确率\t",acc_100per/1000)
				loss_100per = 0
				acc_100per = 0
		print("总计 \n",file = log_file_handle)
		print ("%d \t %.6f \t %.6f\n"
			% (i,
			loss_total/i,
			acc_total/i),
			file=log_file_handle,
			end="")
		log_file_handle.close()

def LSTMtrain():
	print ("LSTMtrain()")
	dir = "../Z_bysj/data/"
	x,y,l=L.database(dir+'protein_transgraph_train',300)
	#x,y,l=L.database(dir+'test',300)
	#print(l)
	epoch = 0
	
	with tf.Session() as sess:
		#件模型
		LSTM = create_RNN_model(sess)
		#记录步数
		global step
		from LSTMclass import step
		#打开log句柄
		log_file = LSTM.log_dir+'log_protein'
		log_file_handle = open(log_file, 'a')
		#保存文件路径
		checkpoint_path = os.path.join(LSTM.model_dir, "protein.ckpt")
		
		losses = []
		while True:
			#规定训练轮数 大于训练轮数便停止
			epoch_last = epoch
			epoch = (step * deep_AS_config.FLAGS.batch_size // l)
			
			train_cycle(LSTM,sess,epoch,checkpoint_path,x,y,l,log_file_handle,losses)

			iftest = epoch-epoch_last
			if iftest:
				
				print("epoch %d : have a test on test-data" %epoch)
				testLSTM(dir+'protein_transgraph_test',LSTM,sess,log_file_handle)
			if epoch >= deep_AS_config.FLAGS.epoch:
				print("EPOCH：%d STOP TRAINING LOOP" %epoch)
				break
		plt.plot(losses)
		plt.savefig("../Z_bysj/log/losses.pdf")
		log_file_handle.close()
		#print (lenth)
		#input_feed[self.input_dict["data1_x"]] = input_x
		#input_feed[self.input_dict["data1_y"]] = input_y
		#loss = tf.reduce_mean(self.loss)correct_pred = tf.equal(tf.argmax(self.output, 1),tf.argmax(self.input_dict["data1_y"], 1))
		#accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
def testLSTM(dir,LSTM,sess,log_file_handle):
	print('testLSTM()')
	print('='*80)
	x,y,l = L.database(dir,deep_AS_config.FLAGS.maxlen)
	statistic={}
	statistic['0->1'] = 0 ##把0预测成1 以此类推
	statistic['0->0'] = 0
	statistic['1->0'] = 0
	statistic['1->1'] = 0
	AUC_x = []
	for i in range(l//1000):
		pre = LSTM.teststep(sess,x[i*1000:(i+1)*1000])
		pre = list(pre)
		for i, element in enumerate(pre):
			pre[i] = list(pre[i])
			if y[i].index(1):
				if pre[i].index(max(pre[i])):
					statistic['1->1'] += 1
				else:
					statistic['1->0'] += 1
			else:
				if pre[i].index(max(pre[i])):
					statistic['0->1'] += 1
				else:
					statistic['0->0'] += 1
	auc_x,auc_y = L.getbatch(x,y,l,1000)
	pre = LSTM.teststep(sess,auc_x)
	pre = list(pre)
	for i, element in enumerate(pre):
		pre[i] = list(pre[i])
		AUC_x.append(pre[i][auc_y[i].index(1)])
	str1 = ','.join(str(e) for e in AUC_x)
	print (statistic)
	print (statistic,file=log_file_handle)
	print (str1,file = log_file_handle)
	log_file_handle.flush()
