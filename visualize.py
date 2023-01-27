#!/usr/bin/env python

"""visualize.py: Implementation of Grad-CAM and Saliency Map."""
__author__      = "David Bertoldi"
__email__       = "d.bertoldi@campus.unimib.it"

import argparse
import utils
import models
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tensorflow.keras.models import Model
from processing import Processing
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap, LinearSegmentedColormap





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
	
			# Get the gradients of the loss w.r.t to the input image
			grads = tape.gradient(loss, convOutputs)

			castConvOutputs = tf.cast(convOutputs > 0, "float32")
			castGrads = tf.cast(grads > 0, "float32")
			guidedGrads = castConvOutputs * castGrads * grads
			convOutputs = convOutputs[0]
			guidedGrads = guidedGrads[0]

			# take average across channels
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
		image = np.uint8(image)
		
		output = cv2.addWeighted(image, alpha, jet_heatmap, 1 - alpha, 0)
		return (jet_heatmap, output)


class SaliencyMap:

	def get_saliency_map(self, model, image, class_idx):
		
		with tf.GradientTape() as tape:
			inputs =  tf.Variable(image, dtype=float)
			tape.watch(inputs)
			predictions = model(inputs)

			loss = predictions[:, class_idx]

			# Get the gradients of the loss w.r.t to the input image.
			gradient = tape.gradient(loss, inputs)
			gradient = tf.nn.relu(gradient)

			# take maximum across channels
			gradient = tf.reduce_max(gradient, axis=-1)

			# convert to numpy
			gradient = gradient.numpy() **0.9

			# normaliz between 0 and 1
			min_val, max_val = np.min(gradient), np.max(gradient)
			smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())

			jet_saliency = np.uint8(cm.jet(smap[0])[..., :3] * 255)
			return jet_saliency


def plot_cm(cm, zero_diagonal=False, labels=None, cmap=plt.cm.viridis):
	"""Plot a confusion matrix."""
	n = len(cm)
	if zero_diagonal:
		for i in range(n):
			cm[i][i] = 0
	size = int(n / 4.)
	fig = plt.figure(figsize=(size, size), dpi=80, )
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	if labels is None:
		labels = [i+1 for i in range(len(cm))]
	x = [i for i in range(len(cm))]
	plt.xticks(x, labels, rotation='vertical')
	y = [i for i in range(len(cm))]
	plt.yticks(y, labels)  # , rotation='vertical'
	res = ax.imshow(np.array(cm), cmap=cmap, interpolation='nearest')
	width, height = cm.shape

	
	plt.colorbar(res)

	plt.show()

if __name__ == '__main__':
	args = parse_arguments()
	config = utils.read_configuration(args.config)

	preprocess_input = keras.applications.xception.preprocess_input
	decode_predictions = keras.applications.xception.decode_predictions


	# Choose model
	class_model = models.Efficientnetb4(0.5)
	model = class_model.get_model()
	model.load_weights('output/checkpoints/{}-loss.h5'.format(type(class_model).__name__.lower()))
	
	preprocessor = class_model.preprocess()

	target_size = class_model.size

	# Load image
	p = Processing(target_size=target_size,
                        batch_size=1,
                        config=config,
                        preprocessor=preprocessor)
	
	test_preprocessed  = p.from_folder('./image')
	image, label = test_preprocessed.__getitem__(0)

	
	img = image[0]
	
	# Print top 4 predictions
	top = 4
	predictions = model.predict(image)[0]
	indexes = np.argpartition(predictions, -top)[-top:]
	indexes = indexes[np.argsort(predictions[indexes])]
	for i in indexes:
		print('{}({}): {}'.format(utils.LABELS[i], i, predictions[i]))

	# Grad-CAM computation
	icam = GradCAM(model, np.argmax(predictions), None)
	heatmap = icam.compute_heatmap(image)
	heatmap = cv2.resize(heatmap, (target_size, target_size))

	# Saliency Map computation
	smap = SaliencyMap().get_saliency_map(model, tf.expand_dims(img, axis=0), np.argmax(predictions)) 
	(heatmap, output) = icam.overlay_heatmap(heatmap, img, 0.4)

	# Plot of Grad-CAM
	fig, ax = plt.subplots(1, 3)
	fig.set_size_inches(20,20)
	ax[0].imshow(heatmap)
	ax[1].imshow(np.uint8(img))
	ax[2].imshow(output)
	plt.show()

	# Plot of Saliency Map
	fig, axes = plt.subplots(1,2,figsize=(14,5))
	axes[0].imshow(np.uint8(img))
	i = axes[1].imshow(smap, alpha=1.0, cmap='jet')
	fig.colorbar(i)
	plt.show()


	# Confusion Matrix
	p = Processing(target_size=target_size,
                        batch_size=16,
                        config=config,
                        preprocessor=preprocessor)

	_, _, test_preprocessed  = p.get_dataset()


	# confusion matrices
	viridis = matplotlib.cm.get_cmap('viridis', 330)
	newcolors = viridis(np.linspace(0, 1, 330))
	white = np.array([1, 1, 1, 1])
	newcolors[:5, :] = white
	newcmp = ListedColormap(newcolors)

	Y_test_pred = model.predict(test_preprocessed)
	y_test_pred = Y_test_pred.argmax(1)
	cm = confusion_matrix(test_preprocessed.classes, y_test_pred)
	plot_cm(cm, cmap=newcmp)
	print(classification_report(y_test_pred, test_preprocessed.classes))
