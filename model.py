import tensorflow as tf
from tensorflow.data import Iterator
from cocob_optimizer import COCOB
from load_dataset import *
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

slim = tf.contrib.slim

class DCGAN:
	def __init__(self, lr_G=0.001, lr_D=0.0002, max_epoch=30, batch_size=32, z_dim=100, is_training=True):
		self.batch_size = batch_size
		self.z_dim = z_dim
		self.tr_dataset = get_dataset(['train.tfrecords'], batch_size=batch_size, num_epoch=max_epoch)
		self.val_dataset = get_dataset(['val.tfrecords'], batch_size=batch_size, num_epoch=max_epoch)
		self.tr_iterator = self.tr_dataset.make_initializable_iterator()
		self.val_iterator = self.val_dataset.make_initializable_iterator()
		self.is_training = is_training
		self.global_step = None


	def build_noise(self):
		random_z = tf.random_uniform([self.batch_size, 1, 1, self.z_dim], -1, 1)
		return random_z


	def generator(self, random_z, is_training=True, reuse=False):
		with tf.variable_scope('Generator', reuse=reuse) as scope:
			batch_norm_params = {'decay': 0.9, 'epsilon': 0.001, 'is_training': is_training, 'scope': 'batch_norm'}

			with slim.arg_scope([slim.conv2d_transpose], 
								kernel_size=[4, 4], 
								stride=[2, 2], 
								activation_fn=tf.nn.leaky_relu, 
								normalizer_fn=slim.batch_norm, 
								normalizer_params=batch_norm_params):

				# random_z: 1 * 1 * 100 dim
				# output: 3 * 3 * 256 dim

				layer1 = slim.fully_connected(random_z, 3*3*256, activation_fn=tf.nn.leaky_relu, scope='layer1')
				proj1 = tf.reshape(layer1, [-1, 3, 3, 256])

				#output: 7 * 7 * 128 dim
				layer2 = slim.conv2d_transpose(proj1, 128, kernel_size=[3, 3], padding='VALID', scope='layer2')

				#output: 14 * 14 * 64 dim
				layer3 = slim.conv2d_transpose(layer2, 64, scope='layer3')

				#output: 28 * 28 * 3 dim
				layer4 = slim.conv2d_transpose(layer3, 3, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='layer4')
				logits = layer4

				return logits


	def discriminator(self, data, is_training=True, reuse=False):
		with tf.variable_scope('Discriminator', reuse=reuse) as scope:
			batch_norm_params = {'decay': 0.9, 'epsilon': 0.001, 'is_training': is_training, 'scope': 'batch_norm'}

			with slim.arg_scope([slim.conv2d], kernel_size=[4, 4], stride=[2, 2], activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
				#data: 28 * 28 * 3 dim
				#output: 14 * 14 * 64 dim
				layer1 = slim.conv2d(data, 64, normalizer_fn=None, scope='layer1')

				#output: 7 * 7 * 128 dim
				layer2 = slim.conv2d(layer1, 128, scope='layer2')

				#output: 3 * 3 * 256 dim
				layer3 = slim.conv2d(layer2, 256, kernel_size=[3, 3], padding='VALID', scope='layer3')

				#output: 1 * 1 * 1 dim
				#activation_fn = None since we'ill use sigmoid cross entropy
				layer4 = slim.conv2d(layer3, 1, kernel_size=[3, 3], stride=[1, 1], padding='VALID', activation_fn=None, scope='layer4')

				#logits: 1 dim, which is [0, 1]
				logits = tf.squeeze(layer4, axis=[1, 2])

				return logits


	def set_global_step(self):
		if self.is_training:
			self.global_step = tf.train.get_or_create_global_step()


	def gan_loss(self, logits, is_real=True, scope=None):

		if is_real:
			label = tf.ones_like(logits)
		else:
			label = tf.zeros_like(logits)

		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=label, logits=logits, scope=scope)
		return loss


	def build_model(self):
		self.set_global_step()

		if not self.is_training:
			return "VAL"

		random_z = self.build_noise()

		# set up data
		real_data = self.tr_iterator.get_next()
		gen_data = self.generator(random_z)

		# get D's logits
		real_logits = self.discriminator(real_data)
		fake_logits = self.discriminator(gen_data, reuse=True)

		# get loss
		loss_real = self.gan_loss(real_logits, scope='loss_real')
		loss_fake = self.gan_loss(fake_logits, is_real=False, scope='loss_fake')

		with tf.variable_scope('loss_D'):
			self.loss_D = loss_real + loss_fake
		self.loss_G = self.gan_loss(fake_logits, is_real=True, scope='loss_G')  # logit: fake & label: real(1) --> D should believe that gen_data is real(1)

		self.var_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
		self.var_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

		# write summaries
		tf.summary.scalar('losses/loss_D', self.loss_D)
		tf.summary.scalar('losses/loss_G', self.loss_G)

		# Add image summaries
		tf.summary.image('gen_images', gen_data, max_outputs=4)
		tf.summary.image('real_images', real_data)

		print('built successfully')


	def print_sample(sample_size):
		sample_z = tf.random_uniform([sample_size, 1, 1, self.z_dim], -1, 1)
		sample_gen_data = self.generator(sample_z, is_training=False, reuse=True)

		image = sample_gen_data[:sample_size, :, :]
		image = image.reshape([sample_size, 28, 28, 3])
		image = image.swapaxes(0, 1)
		image = image.reshape([28, sample_size*28*3])

		plt.figure(figsize=(sample_size, 1))
		plt.axis('off')
		plt.imshow(print_images)
		plt.show()