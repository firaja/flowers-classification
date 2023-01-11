import keras
import utils
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import random
import math



TF_AUTOTUNE = tf.data.AUTOTUNE


class Processing:

    def __init__(self, target_size, batch_size, augment=True, shuffle=True, brightness_delta=0.3, flip=True, rotation=20, preprocessor=None):
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.brightness_delta = brightness_delta
        self.flip = flip
        self.rotation = rotation
        self.preprocessor = preprocessor


    def get_dataset(self):
        (ds_train, ds_validation, ds_test) = self.get_dataset_eagerly()

        
        train_preprocessed = ds_train.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE)
        train_preprocessed = train_preprocessed.map(lambda x, y: (self.__augment__(x), y), num_parallel_calls=TF_AUTOTUNE).repeat(count=6).batch(self.batch_size).cache().prefetch(TF_AUTOTUNE)
        

        test_preprocessed = ds_test.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE)
        test_preprocessed = test_preprocessed.map(lambda x, y: (self.__augment__(x, augment=False), y), num_parallel_calls=TF_AUTOTUNE).batch(self.batch_size).cache().repeat().prefetch(TF_AUTOTUNE)

        validation_preprocessed = ds_validation.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE)
        validation_preprocessed = validation_preprocessed.map(lambda x, y: (self.__augment__(x, augment=False), y), num_parallel_calls=TF_AUTOTUNE).batch(self.batch_size).cache().repeat().prefetch(TF_AUTOTUNE)

        train_cardinality = ds_train.cardinality().numpy()
        validation_cardinality = ds_validation.cardinality().numpy()

        train_preprocessed = train_preprocessed.shuffle(train_cardinality)
        validation_preprocessed = validation_preprocessed.shuffle(validation_cardinality)

        return train_preprocessed, test_preprocessed, validation_preprocessed, train_cardinality, validation_cardinality

    def get_dataset_eagerly(self):
        splits = ['train', 'validation', 'test']
        splits, ds_info = tfds.load('oxford_flowers102', split=splits, with_info=True)
        return splits


    def __parse_image__(self, features):
        image = features['image']
        image = tf.image.resize(image, (self.target_size, self.target_size)) // 255
        return image, features['label']

    def __augment__(self, original, augment=True):
        image = original
        if augment:
            image = tf.image.random_brightness(image, self.brightness_delta)
            image = tf.image.random_flip_left_right(image)
            image = tfa.image.transform_ops.rotate(image, random.randrange(-self.rotation, self.rotation))
        if self.preprocessor:
            1#image = self.preprocessor(image)
        return image
