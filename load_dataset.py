import tensorflow as tf

def parser(record):
	feature = {'image_raw': tf.FixedLenFeature([], tf.string)}
	parsed = tf.parse_single_example(record, feature)
	image = tf.decode_raw(parsed["image_raw"], tf.uint8)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, shape=[28, 28, 3])
	return image

def get_dataset(filenames, batch_size=32, num_epoch=30):
	dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
	# dataset = dataset.repeat(num_epoch)
	dataset = dataset.apply(
		tf.contrib.data.map_and_batch(parser, batch_size)
	)
	dataset = dataset.prefetch(buffer_size=2)
	return dataset
