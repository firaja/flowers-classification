import keras
from keras_applications.resnext import preprocess_input
from keras import preprocessing




def pre_processing(image):
    return preprocess_input(image, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

def train_data_generator():
    return preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        rotation_range=10,
        preprocessing_function=pre_processing)


def test_data_generator():
    return preprocessing.image.ImageDataGenerator(preprocessing_function=pre_processing)