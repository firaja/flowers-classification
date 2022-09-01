import keras
import utils
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import random
import math



TF_AUTOTUNE = tf.data.AUTOTUNE


class Processing:

    def __init__(self, target_size, batch_size, shuffle=True, brightness_delta=0.3, flip=True, rotation=10):
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.brightness_delta = brightness_delta
        self.flip = flip
        self.rotation = math.floor(rotation * (math.pi / 180))


    def get_dataset(self):
        splits = ['test', 'validation', 'train']
        splits, ds_info = tfds.load('oxford_flowers102', split=splits, with_info=True)
        (ds_train, ds_validation, ds_test) = splits

        train_cardinality = ds_train.cardinality().numpy()
        validation_cardinality = ds_validation.cardinality().numpy()

        # Train Datastore
        train_preprocessed = ds_train

        if self.shuffle:
            train_preprocessed = ds_train.shuffle(train_cardinality)
        
        train_preprocessed = train_preprocessed.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE)

        if self.brightness_delta > 0:
            train_preprocessed = train_preprocessed.map(lambda image, label: (tf.image.random_brightness(image, self.brightness_delta), label))

        if self.flip:
            train_preprocessed = train_preprocessed.map(lambda image, label: (tf.image.random_flip_left_right(image), label))

        if self.rotation > 0:
            train_preprocessed = train_preprocessed.map(lambda image, label: (tfa.image.transform_ops.rotate(image, random.randrange(-self.rotation, self.rotation)), label))


        train_preprocessed = train_preprocessed.batch(self.batch_size).cache().prefetch(TF_AUTOTUNE)

        test_preprocessed = ds_test.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE).cache().batch(self.batch_size).prefetch(TF_AUTOTUNE)

        validation_preprocessed = ds_validation.map(self.__parse_image__, num_parallel_calls=TF_AUTOTUNE).cache().batch(self.batch_size).prefetch(TF_AUTOTUNE)


        return train_preprocessed, test_preprocessed, validation_preprocessed, train_cardinality, validation_cardinality


    def __parse_image__(self, features):
        image = features['image']
        image = tf.image.resize(image, (self.target_size, self.target_size)) / 255.0
        return image, features['label']