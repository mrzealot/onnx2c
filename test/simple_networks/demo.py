# Trivial network generator
# This is intended to generate onnx models
# used as tests in onnx2c.
# NB: this script does not train the model at all
# and all weights are left at 1.0. This is so it
# is possible to manually calculate the correct output
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras2onnx


if __name__ == "__main__":

	x = np.array( [
		[ 1,  2,  3,  4,  5],
		[ 6,  7,  8,  9, 10],
		[11, 12, 13, 14, 15],
		[16, 17, 18, 19, 20],
		[21, 22, 23, 24, 25] ]).astype(np.float32)
	x = x[np.newaxis, ... ]
	x = x[..., np.newaxis]


	model = tf.keras.Sequential()
	model.add(keras.layers.Conv2D(1, (2, 2), strides=2,
		kernel_initializer="ones"
		))
	#model.add(keras.layers.MaxPooling2D((2, 2), strides=2))

	# flatten to 1d makes it easier to have a generic main()
	model.add(keras.layers.Flatten())

	optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
	model.compile(optimizer=optimizer, loss="MSE")

	predictions = model.predict(x)

	print(predictions)

	onnx_model = keras2onnx.convert_keras(model, "demo")
	keras2onnx.save_model(onnx_model, "demo.onnx")
