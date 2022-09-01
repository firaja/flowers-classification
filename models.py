import tensorflow.keras
from classification_models.keras import Classifiers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import DenseNet121, InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation, Conv2D, MaxPool2D, GlobalMaxPool2D
from keras_applications.resnext import ResNeXt50, preprocess_input
from keras_efficientnets import EfficientNetB4
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.vgg16 import VGG16
import sys, inspect


WEIGHTS='imagenet'
ACTIVATION = 'softmax'


ARCHITECTURES = {}

OPTIMIZERS = {
    'Adam': {'get': lambda : lambda : Adam(learning_rate=0.001)}, 
    'SGD': {'get': lambda : lambda : SGD(learning_rate=0.001, momentum=0.9)}
    }

def last_conv(model):
    return list(filter(lambda x: isinstance(x, Conv2D), model.layers))[-1].name
    


class Vgg16 :

    size = 224

    def __init__(self, dropout):
        vgg16 = VGG16(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        output = GlobalAveragePooling2D()(vgg16.output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=vgg16.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)


class Efficientnetb4:

    size = 380

    def __init__(self, dropout):
        efficientnet = EfficientNetB4(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        output = GlobalAveragePooling2D()(efficientnet.output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=efficientnet.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)
    

class Resnet18:

    size = 224

    def __init__(self, dropout):
        resnet, _ = Classifiers.get('resnet18')
        base_model = resnet(input_shape=(self.size, self.size, 3), include_top=False, weights=WEIGHTS)
        
        output = GlobalAveragePooling2D()(base_model.output)
        output = Dropout(dropout)(output)
        output = Dense(102, activation=ACTIVATION)(output)
        output = Activation(ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=base_model.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)

class Resnet50:
    size = 224

    def __init__(self, dropout):
        resnet = ResNet50(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        output = GlobalAveragePooling2D()(resnet.output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=resnet.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)


class Inceptionv3:

    size = 224

    def __init__(self, dropout):
        inceptionV3 = InceptionV3(weights=WEIGHTS, include_top=False)

        output = GlobalAveragePooling2D()(inceptionV3.output)
        output = Dropout(dropout)(output)
        output = Dense(102)(output)
        output = Activation(ACTIVATION, dtype='float32', name='predictions')(output)
        self.model = Model(inputs=inceptionV3.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)

# Populate architectures
current_module = sys.modules[__name__]
for name, obj in inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__):
    ARCHITECTURES[name.lower()] = obj