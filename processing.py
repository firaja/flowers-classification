#!/usr/bin/env python

"""processing.py: Implementation of the Cyclical Learning Rate Finder algorithm."""
__author__      = "David Bertoldi"
__email__       = "d.bertoldi@campus.unimib.it"

import utils
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import random
import math



TF_AUTOTUNE = tf.data.AUTOTUNE


class Processing:

    def __init__(self, target_size, batch_size, config, brightness_delta=0.3, flip=True, rotation=20, zoom_delta=0.2, preprocessor=None):
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=flip,
            brightness_range=[1-brightness_delta, 1+brightness_delta],
            rotation_range=rotation,
            zoom_range=[1-zoom_delta, 1+zoom_delta],
            preprocessing_function=preprocessor)

        self.test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessor)

        self.config=config
        self.target_size=target_size
        self.batch_size=batch_size


    def get_dataset(self):
        """ Loads train, validation and test datasets. Only on the training datasets are applied transformations
        """
        train_generator = self.train_datagen.flow_from_directory(directory=self.config['paths']['data']['training'],
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            target_size=(self.target_size, self.target_size),
                                                            class_mode=self.config['training']['mode'])


        valid_generator = self.test_datagen.flow_from_directory(directory=self.config['paths']['data']['validation'],
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            target_size=(self.target_size, self.target_size),
                                                            class_mode=self.config['training']['mode'])

        test_generator = self.test_datagen.flow_from_directory(directory=self.config['paths']['data']['test'],
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            target_size=(self.target_size, self.target_size),
                                                            class_mode=self.config['training']['mode'])


        return train_generator, valid_generator, test_generator

    def from_folder(self, path):
        """ Loads images from a folder without transformations
        """
        return self.test_datagen.flow_from_directory(directory=path,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            target_size=(self.target_size, self.target_size),
                                                            class_mode=self.config['training']['mode'])