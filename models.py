import keras
from classification_models.keras import Classifiers
from keras import Sequential, Model
from keras.applications import DenseNet121
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from keras_applications.resnext import ResNeXt50, preprocess_input
from keras_efficientnets import EfficientNetB4
from keras import backend as K
from keras.optimizers import Adam, SGD


WEIGHTS='imagenet'
ACTIVATION = 'softmax'


ARCHITECTURES = {
    'densenet121': {'size': 224, 'get': lambda d : lambda : efficientnetb4(d)}, 
    'efficientnetb4': {'size': 380, 'get': lambda d : lambda : efficientnetb4(d)}, 
    'resnet18': {'size': 224, 'get': lambda d : lambda : resnet18(d)}
    }

OPTIMIZERS = {
    'Adam': {'get': lambda : lambda : Adam(learning_rate=0.01)}, 
    'SGD': {'get': lambda : lambda : SGD(learning_rate=0.01, momentum=0.9)}
    }


def densenet121(dropout):
    densenet = DenseNet121(weights=WEIGHTS, include_top=False, input_shape=(224, 224, 3))

    output = GlobalAveragePooling2D()(densenet.output)
    output = Dropout(dropout)(output)
    output = Dense(102, activation=ACTIVATION)(output)

    return Model(densenet.input, output)



def efficientnetb4(dropout):
    efficientnet = EfficientNetB4(weights=WEIGHTS, include_top=False, input_shape=(380, 380, 3))

    model = Sequential()
    model.add(efficientnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout))
    model.add(Dense(102, activation=ACTIVATION))
    return model


def resnet18(dropout):
    resnet, _ = Classifiers.get('resnet18')
    base_model = resnet(input_shape=(224, 224, 3), include_top=False, weights=WEIGHTS)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(102, activation=ACTIVATION)(x)
    return Model(inputs=[base_model.input], outputs=[output])