import tensorflow as tf
from glob import glob
import cv2
import sys

DATA_FILE = glob('data/img_align_celeba/*')
NUM_DATA = len(DATA_FILE)
NUM_VAL = NUM_DATA // 10
IMG_WIDTH = 178
IMG_HEIGHT = 218 
RESIZE_WIDTH = RESIZE_HEIGHT = 28  # the width and height of images that will be resized. 

# define wrapper for feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(path):
	img = cv2.imread(path)
	residual = (IMG_HEIGHT - IMG_WIDTH) // 2
	img_cropped = img[residual:residual+IMG_WIDTH, :]  # crop the center of images. now [178, 178]
	img_resized = cv2.resize(img_cropped, (RESIZE_WIDTH, RESIZE_HEIGHT))  # resize to [28, 28] to make training easier
	img_colored = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # default cv color is BGR
	img_normed = img_colored / 256
	return img_normed

def create_tf_record(filename, paths):
	writer = tf.python_io.TFRecordWriter(filename)

	print('Saving: %s...' % filename)

	for i in range(len(paths)):
		if i % 1000 == 0:
			print('Saving: %d / %d' % (i, len(paths)))
			sys.stdout.flush()  # buffering improves IO performance

		img = load_image(paths[i])  # numpy array

		feature = {'image_raw': _bytes_feature(img.tostring())}
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# write on the file
		writer.write(example.SerializeToString())
	writer.close()
	sys.stdout.flush()

	print('Saving done: %s' % filename)

# don't have to shuffle. We will load face images.
val_dataset = DATA_FILE[:NUM_VAL]
tr_dataset = DATA_FILE[NUM_VAL:]

create_tf_record('train.tfrecords', tr_dataset)
create_tf_record('val.tfrecords', val_dataset)