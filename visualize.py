import argparse
import utils
import models
import processing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tensorflow.keras.models import Model



def parse_arguments():
	parser = argparse.ArgumentParser(description='Flower Recognition Visualization')
	parser.add_argument('--config', type=str, const='config.yml', default='config.yml', nargs='?', help='Configuration file')
	
	return parser.parse_args()

class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4:
				return layer.name

		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


	def compute_heatmap(self, image, eps=1e-8):
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, self.model.output])

		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			
			loss = predictions[:, tf.argmax(predictions[0])]
	
			grads = tape.gradient(loss, convOutputs)

			castConvOutputs = tf.cast(convOutputs > 0, "float32")
			castGrads = tf.cast(grads > 0, "float32")
			guidedGrads = castConvOutputs * castGrads * grads
			convOutputs = convOutputs[0]
			guidedGrads = guidedGrads[0]

			weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
			cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

			(w, h) = (image.shape[2], image.shape[1])
			heatmap = cv2.resize(cam.numpy(), (w, h))

			numer = heatmap - np.min(heatmap)
			denom = (heatmap.max() - heatmap.min()) + eps
			heatmap = numer / denom
			heatmap = (heatmap * 255).astype("uint8")

			return heatmap

	def overlay_heatmap(self, heatmap, image, alpha):
		
		jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)
		heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
		output = cv2.addWeighted(image, alpha, jet_heatmap, 1 - alpha, 0)
		return (jet_heatmap, output)


if __name__ == '__main__':
	args = parse_arguments()
	config = utils.read_configuration(args.config)

	preprocess_input = keras.applications.xception.preprocess_input
	decode_predictions = keras.applications.xception.decode_predictions

	class_model = models.Resnet18(0.5)
	model = class_model.get_model()
	model.load_weights('output/checkpoints/{}-loss.h5'.format(type(class_model).__name__.lower()))
	model.layers[-1].activation = None
	model.summary()
	target_size = class_model.size
	


	img_path = utils.get_path('/home/firaja/Downloads/moon.jpg')
	image = cv2.imread(img_path)	

	image = cv2.resize(image, (target_size, target_size))
	image = tf.expand_dims(image, axis=-1)
	image = tf.divide(image, 255)
	image = tf.reshape(image, [1, target_size, target_size, 3])

	top = 4
	predictions = model.predict(image)[0]
	#for i in range(len(predictions)):
	#	print('{}({}): {}'.format(utils.LABELS[i], i+1, predictions[i]))
	indexes = np.argpartition(predictions, -top)[-top:]
	indexes = indexes[np.argsort(predictions[indexes])]
	print('---------------')
	for i in indexes:
		print('{}({}): {}'.format(utils.LABELS[i], i+1, predictions[i]))

	icam = GradCAM(model, np.argmax(predictions), None) 
	heatmap = icam.compute_heatmap(image)
	heatmap = cv2.resize(heatmap, (target_size, target_size))

	image = cv2.imread(img_path)
	image = cv2.resize(image, (target_size, target_size))

	(heatmap, output) = icam.overlay_heatmap(heatmap, image, 0.3)


	fig, ax = plt.subplots(1, 3)
	fig.set_size_inches(20,20)

	ax[0].imshow(heatmap)
	ax[1].imshow(image)
	ax[2].imshow(output)
	plt.show()