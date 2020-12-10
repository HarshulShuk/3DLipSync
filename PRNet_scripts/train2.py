
import tensorflow.keras.layers as L
import tensorflow as tf
import tensorflow
from skimage.transform import estimate_transform, warp
from PIL import Image
from random import uniform, randint
import os
import dlib
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training PRNet')
parser.add_argument('--weights_path', type=str, help='Path to weights', required=True)
args = parser.parse_args()


tf.enable_eager_execution()
prefix = "/mnt/c/Users/Harshul/Desktop/F2020/PRNet"
batch_size = 16
input_size = 256
output_size = 256


# Setup face detector
detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
face_detector = dlib.cnn_face_detection_model_v1(detector_path)


def get_label(img_path):
	npy_path = img_path.replace('jpg', 'npy')
	return tf.convert_to_tensor(np.load(npy_path))


def decode_img(img, file_path):
	img = tf.image.decode_jpeg(img, channels=3).numpy()

	# Scale
	img = np.clip(img * uniform(.6,1.4), 0,255)
	# Rotate
	img = np.array(Image.fromarray(np.uint8(img)).rotate(randint(-45,45)))

	# Detect face
	detected_faces = face_detector(img,1)
	if len(detected_faces) == 0:
	    print('warning: no detected face')
	    print(file_path)
	    exit()
	    return None

	d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
	left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
	old_size = (right - left + bottom - top)/2
	center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
	size = int(old_size*1.58)

	# Crop image
	src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
	DST_PTS = np.array([[0,0], [0, input_size - 1], [input_size - 1, 0]])
	tform = estimate_transform('similarity', src_pts, DST_PTS)
	img = img/255.
	cropped_image = warp(img, tform.inverse, output_shape=(input_size, input_size))

	#img = tf.image.decode_jpeg(cropped_image, channels=3)
	#img = tf.convert_to_tensor(cropped_image)
	img = tf.convert_to_tensor(cropped_image)
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	# resize the image to the desired size.
	return tf.image.resize(img, [input_size, input_size])

def process_path(file_path):
	label = get_label(file_path)
	img = tf.io.read_file(file_path)
	img = decode_img(img, file_path)
	return img, label


shfl_seed = 12312345
tr_size = .8
data_path = "/mnt/c/Users/Harshul/Downloads/300W-LP/300W_LP/AFW/results/*.jpg"
full_ds = tf.data.Dataset.list_files(data_path)
#full_ds.shuffle(num_rows, seed=shfl_seed)

num_rows = len((list(full_ds)))
num_train = int(float(num_rows) * tr_size)
tr_ds = full_ds.take(num_train)
val_ds = full_ds.skip(num_train)

tr_ds = tr_ds.map(lambda file: tf.py_func(process_path, [file], (tf.float32, tf.float32)))
val_ds = val_ds.map(lambda file: tf.py_func(process_path, [file], (tf.float32, tf.float32)))
tr_ds = tr_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)



tf.disable_eager_execution()
tf.disable_v2_behavior()


class MyResBlock(tf.keras.layers.Layer):
	def __init__(self, num_outputs, kernel_size, stride, activation_fn=L.ReLU, normalizer_fn=L.BatchNormalization, scope=None):
		super(MyResBlock, self).__init__()
		self.num_outputs = num_outputs
		self.kernel_size = kernel_size
		self.stride = stride
		self.activation = activation_fn()
		self.normalizer = normalizer_fn(axis=1)

	def build(self, input_shape):
		self.shortcutConv = L.Conv2DTranspose(filters=self.num_outputs, kernel_size=1, strides=1, activation=None)
		self.conv1 = L.Conv2D(filters=self.num_outputs/2, kernel_size=1, strides=1, padding="SAME")
		self.conv2 = L.Conv2D(filters=self.num_outputs/2, kernel_size=self.kernel_size, strides=1, padding="SAME")
		self.conv3 = L.Conv2D(filters=self.num_outputs, kernel_size=1, strides=1, padding="SAME")


	def call(self, x):
		shortcut = x
		if self.stride != 1 or x.get_shape()[3] != num_outputs:
			shortcut = self.shortcutConv(shortcut)

		x = self.conv3(self.conv2(self.conv1(x)))
		x += shortcut
		x = self.activation(self.normalizer(x))
		return x



size = 16
model = tf.keras.Sequential([
	L.Conv2D(filters=size, kernel_size=4, strides=1),
	MyResBlock(size * 2, 4, 2),
	MyResBlock(size * 2, 4, 1),
	MyResBlock(size * 4, 4, 2),
	MyResBlock(size * 4, 4, 1),
	MyResBlock(size * 8, 4, 2),
	MyResBlock(size * 8, 4, 1),
	MyResBlock(size * 16, 4, 2),
	MyResBlock(size * 16, 4, 1),
	MyResBlock(size * 32, 4, 2),
	MyResBlock(size * 32, 4, 1),


	L.Conv2DTranspose(filters=size*32,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=size*16,  kernel_size=4, strides=2),
	L.Conv2DTranspose(filters=size*16,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=size*16,  kernel_size=4, strides=1),

	L.Conv2DTranspose(filters=size*8,  kernel_size=4, strides=2),
	L.Conv2DTranspose(filters=size*8,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=size*8,  kernel_size=4, strides=1),

	L.Conv2DTranspose(filters=size*4,  kernel_size=4, strides=2),
	L.Conv2DTranspose(filters=size*4,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=size*4,  kernel_size=4, strides=1),

	L.Conv2DTranspose(filters=size*2,  kernel_size=4, strides=2),
	L.Conv2DTranspose(filters=size*2,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=size,  kernel_size=4, strides=2),
	L.Conv2DTranspose(filters=size,  kernel_size=4, strides=1),

	L.Conv2DTranspose(filters=3,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=3,  kernel_size=4, strides=1),
	L.Conv2DTranspose(filters=3,  kernel_size=4, strides=1, activation=tf.keras.activations.sigmoid),
])


new_mask = np.asarray(Image.open("new_uv_weight_mask.jpg").convert("L"))
new_mask = new_mask/255.0
def w_MSE(y, yPred):
	global new_mask
	diff = (y - yPred) ** 2
	return (diff * new_mask).mean()


opt = tensorflow.keras.optimizers.Adam(learning_rate=.0001)
model.compile(optimizer=opt, loss=w_MSE, metrics=[tf.keras.metrics.MeanSquaredError()])



if not os.path.isfile(args.weights_path):
    print("please download PRN trained model first.")
    exit()
model.load_weights(args.weights_path)


batch_size = 16
model.fit(x=tr_ds, epochs=10, verbose=2, validation_data=val_ds, validation_steps=None)
# for epoch in range(10):
# 	model.fit(x=tr_ds, batch_size=batch_size)
#     # for x_batch, y_batch in tr_ds
#     #     model.fit(x_batch, y_batch)
#     val_loss = model.evaluate(val_ds, batch_size=batch_size)
#     print("Epoch {} had loss {}".format(epoch, val_loss))
#model.train_on_batch(tr_ds)


model.save_weights(os.path.join(prefix, 'Data/net-data/justweightsckpt'))
model.save(os.path.join(prefix, 'Data/net-data/fullmodel.h5'))