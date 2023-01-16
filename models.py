import tensorflow as tf
import tensorflow.keras
from classification_models.keras import Classifiers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import DenseNet121, InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation, Conv2D, MaxPool2D, GlobalMaxPool2D, MaxPooling2D
from keras_applications.resnext import ResNeXt50, preprocess_input
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.vgg16 import VGG16
import sys, inspect


WEIGHTS='imagenet'
ACTIVATION = 'relu'
FINAL_ACTIVATION = 'softmax'

ARCHITECTURES = {}

OPTIMIZERS = {
    'Adam': {
        'get': lambda : lambda : Adam(learning_rate=1e-6),
        'lr': [1e-6, 5e-4]
    }, 
    'SGD': {
        'get': lambda : lambda : SGD(learning_rate=0.001, momentum=0.9),
        'lr': [1e-5, 1e-3]
        }
    }

def last_conv(model):
    return list(filter(lambda x: isinstance(x, Conv2D), model.layers))[-1].name
    



class Efficientnetb4:

    size = 224

    def __init__(self, dropout):
        base_model = EfficientNetB4(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(512)(output)
        output = BatchNormalization()(output)
        output = Activation(ACTIVATION)(output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(FINAL_ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=base_model.input, outputs=output)


    def get_model(self):
        return self.model

    def preprocess(self):
        return tf.keras.applications.efficientnet.preprocess_input

    def get_last_conv(self):
        return last_conv(self.model)




class FrozenEfficientnetb4:

    size = 224

    def __init__(self, dropout):
        base_model = EfficientNetB4(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        base_model.trainable = True

        for layer in base_model.layers:
            layer.trainable = False

        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(512)(output)
        output = BatchNormalization()(output)
        output = Activation(ACTIVATION)(output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(FINAL_ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=base_model.input, outputs=output)


    def get_model(self):
        return self.model

    def preprocess(self):
        return tf.keras.applications.efficientnet.preprocess_input

    def get_last_conv(self):
        return last_conv(self.model)
    

class Resnet18:

    size = 224

    def __init__(self, dropout):
        resnet, _ = Classifiers.get('resnet18')
        base_model = resnet(input_shape=(self.size, self.size, 3), include_top=False, weights=WEIGHTS)
        
        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(512)(output)
        output = BatchNormalization()(output)
        output = Activation(ACTIVATION)(output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(FINAL_ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=base_model.input, outputs=output)

    def get_model(self):
        return self.model

    def preprocess(self):
        return tf.keras.applications.resnet50.preprocess_input


    def get_last_conv(self):
        return last_conv(self.model)


class Inceptionv3:

    size = 299

    def __init__(self, dropout):

        base_model = InceptionV3(weights=WEIGHTS, include_top=False)

        output = GlobalAveragePooling2D()(base_model.output)
        output = Dense(512)(output)
        output = BatchNormalization()(output)
        output = Activation(ACTIVATION)(output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(FINAL_ACTIVATION, dtype='float32')(output)
        self.model = Model(inputs=base_model.input, outputs=output)

    def get_model(self):
        return self.model

    def preprocess(self):
        return tf.keras.applications.inception_v3.preprocess_input

    def get_last_conv(self):
        return last_conv(self.model)








# Populate architectures
current_module = sys.modules[__name__]
for name, obj in inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__):
    ARCHITECTURES[name.lower()] = obj