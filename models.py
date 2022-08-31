import tensorflow.keras
from classification_models.keras import Classifiers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from keras_applications.resnext import ResNeXt50, preprocess_input
from keras_efficientnets import EfficientNetB4
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
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
    


class Densenet121 :

    size = 224

    def __init__(self, dropout):
        densenet = DenseNet121(weights=WEIGHTS, include_top=False, input_shape=(self.size, self.size, 3))

        output = GlobalAveragePooling2D()(densenet.output)
        output = Dropout(dropout)(output)
        output = Dense(102, activation=ACTIVATION)(output)
        self.model = Model(inputs=densenet.input, outputs=output)

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
        output = Dense(102, activation=ACTIVATION)(output)
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
        self.model = Model(inputs=base_model.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)

class MobileNet:
    size = 224

    def __init__(self, dropout):
        resnet, _ = Classifiers.get('resnet18')
        base_model = resnet(input_shape=(self.size, self.size, 3), include_top=False, weights=WEIGHTS)
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(dropout))
        model.add(Dense(102, activation=ACTIVATION))
        self.model = model

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)

class Inceptionv3:

    size = 224

    def __init__(self, dropout):
        inceptionV3 = InceptionV3(weights=WEIGHTS, include_top=False)
        inceptionV3.trainable = False
        output = GlobalAveragePooling2D()(inceptionV3.output)
        output = Dense(512)(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(dropout)(output)
        output = Dense(102, activation=ACTIVATION)(output)
        self.model = Model(inputs=inceptionV3.input, outputs=output)

    def get_model(self):
        return self.model

    def get_last_conv(self):
        return last_conv(self.model)

# Populate architectures
current_module = sys.modules[__name__]
for name, obj in inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__):
    ARCHITECTURES[name.lower()] = obj