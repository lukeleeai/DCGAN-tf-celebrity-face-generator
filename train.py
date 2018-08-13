import tensorflow as tf
from model import DCGAN
# from cocob_optimizer import COCOB

train_dir = 'train/dcgan/face_gen/'
k = 1
num_epoch = 30
save_epoch = 5
print_step = 500
summary_step = 2500

learning_rate_D = 0.0002
learning_rate_G = 0.001


if __name__ == '__main__':
	model = DCGAN(batch_size=64)
	model.build_model()

	# set up optimization
	opt_D = tf.train.AdamOptimizer(learning_rate=learning_rate_D, beta1=0.5)
	opt_G = tf.train.AdamOptimizer(learning_rate=learning_rate_G, beta1=0.5)

	collection_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')  # collections of variables in variable_scope('discriminator')
	collection_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

	with tf.control_dependencies(collection_D):
		train_D = opt_D.minimize(loss=model.loss_D, var_list=model.var_D)
	with tf.control_dependencies(collection_G):
		train_G = opt_G.minimize(loss=model.loss_G, var_list=model.var_G, global_step=model.global_step)

	# save graphs
	graph_location = train_dir
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph()) 

	# merge summaries
	summary_op = tf.summary.merge_all()

	# save parameters
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('start training')

		for epoch in range(num_epoch):
			sess.run(model.tr_iterator.initializer)
			while True:
				try:
					for _ in range(k):
						_, loss_D = sess.run([train_D, model.loss_D])
					_, loss_G, global_step = sess.run([train_G, model.loss_G, model.global_step])

					if not global_step % print_step:
						print('Epoch: %d, Step: %d' % (epoch, global_step))
						model.print_sample(10)

					if not global_step % summary_step:
						summary_str = sess.run(summary_op)
						train_writer.add_summary(summary_str, global_step=global_step)

				except tf.errors.OutOfRangeError:
					print('end of dataset')
					break

			if not epoch % save_epoch:
				print('Saving model in %s. Global step: %d' % (train_dir + 'model.ckpt', global_step))
				saver.save(sess, train_dir + 'model.ckpt', global_step=global_step)










