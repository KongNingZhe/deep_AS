import deep_AS_get_batch
import deep_AS_train
import tensorflow as tf
import deep_AS_config 

##训练命令，得到loss(train_step)
class TrainingModel(object):
	def __init__(self, session, training_mode):
		print("TrainingModel: __init__()")
		
		#self.global_step = tf.Variable(0, trainable=False)
		
		# Dropout probabilities
		#self.keep_conv_holder = tf.placeholder(dtype=tf.float32, name="keep_conv")
		#self.keep_dense_holder = tf.placeholder(dtype=tf.float32, name="keep_dense")

		# INPUT PLACEHOLDERS
		self.x = tf.placeholder(tf.float32, [deep_AS_config.batch_size, deep_AS_config.num_units, deep_AS_config.vocab_size],name="x_input")
		self.y = tf.placeholder(tf.int32, [deep_AS_config.batch_size],name="y_input")
		
		# OUTPUTS and LOSSES
		#self.outputs = deep_AS_train.get_logits(self.x)
		self.losses = deep_AS_train.train(self.x,self.y)
		#self.accuracy = deep_AS_train.accuracy(self.x,self.y)


		#  for training the model.
		if training_mode:
			train_step = tf.train.AdamgradOptimizer(learning_rate).minimize(self.losses)
			
		# Saver
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


	def step(self,session,encode_input,training_mode=True):
		
		#
		input_feed = {}
		if training_mode:
			training_losses = []
			acc_list = []
			output_feed = [train_step,self.losses,self.accuracy]
			_, step_loss, acc = session.run(fetches=output_feed, feed_dict=input_feed)
			training_loss+=step_loss
			if step % deepnovo_config.steps_per_checkpoint == 0:
				print('第{0} 步的平均损失 {1}和准确率{2}'.format(step, training_loss/deepnovo_config.steps_per_checkpoint,acc))
				training_loss = 0
					

			else:
					output_feed = [self.losses,self.accuracy]  # Loss for this batch.
					step_loss, acc = session.run(fetches=output_feed, feed_dict=input_feed)
					print (step_loss, acc)
