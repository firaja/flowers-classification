import os
import argparse
import models
import processing
import utils
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from clr_callback import CyclicLR
import tensorflow_datasets as tfds
import tensorflow_hub as hub


np.random.seed(42)
tf.random.set_seed(42)
    


CLRS = ['triangular', 'triangular2', 'exp']

EPOCHS = 50
TF_AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 473

def parse_image(features):
    image = features['image']
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, features['label']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Flower Recognition Neural Network')

    parser.add_argument('--batch', type=int, const=64, default=64, nargs='?', help='Batch size used during training')
    parser.add_argument('--arch', type=str, const='resnet18', default='resnet18', nargs='?', choices=models.ARCHITECTURES.keys(), help='Architecture')
    parser.add_argument('--opt', type=str, const='Adam', default='SGD', nargs='?', choices=models.OPTIMIZERS.keys(), help='Optimizer')
    parser.add_argument('--clr', type=str, const='triangular', default='triangular', nargs='?', choices=CLRS, help='Cyclical learning rate')
    parser.add_argument('--step', type=float, const=8, default=8, nargs='?', help='Step size')
    parser.add_argument('--dropout', type=float, const=0.5, default=0.5, nargs='?', help='Dropout rate')
    parser.add_argument('--config', type=str, const='config.yml', default='config.yml', nargs='?', help='Configuration file')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = utils.read_configuration(args.config)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    architecture = models.ARCHITECTURES[args.arch]
    
    model = architecture(args.dropout).get_model()
    
    target_size = architecture.size


    
    optimizer = models.OPTIMIZERS[args.opt]['get']()()



    splits = ['test', 'validation', 'train']
    splits, ds_info = tfds.load('oxford_flowers102', split=splits, with_info=True)
    (ds_train, ds_validation, ds_test) = splits

    train_preprocessed = ds_train.shuffle(SHUFFLE_BUFFER_SIZE).map(parse_image, num_parallel_calls=TF_AUTOTUNE).cache().batch(args.batch).prefetch(TF_AUTOTUNE)

    test_preprocessed = ds_test.map(parse_image, num_parallel_calls=TF_AUTOTUNE).cache().batch(args.batch).prefetch(TF_AUTOTUNE)

    validation_preprocessed = ds_validation.map(parse_image, num_parallel_calls=TF_AUTOTUNE).cache().batch(args.batch).prefetch(TF_AUTOTUNE)


    train_generator = processing.train_data_generator().flow_from_directory(directory=utils.get_path(config['paths']['train']),
                                                                            batch_size=args.batch,
                                                                            shuffle=True,
                                                                            target_size=(target_size, target_size),
                                                                            interpolation=config['training']['interpolation'],
                                                                            class_mode=config['training']['mode'])

    valid_generator = processing.test_data_generator().flow_from_directory(directory=utils.get_path(config['paths']['valid']),
                                                                            batch_size=args.batch,
                                                                            shuffle=True,
                                                                            target_size=(target_size, target_size),
                                                                            class_mode=config['training']['mode'])

    test_generator = processing.test_data_generator().flow_from_directory(directory=utils.get_path(config['paths']['test']),
                                                                            batch_size=args.batch,
                                                                            shuffle=False,
                                                                            target_size=(target_size, target_size),
                                                                            class_mode=config['training']['mode'])

    model.compile(loss=config['training']['loss'], optimizer=optimizer, metrics=['acc'])

    mcp_save_acc = ModelCheckpoint(utils.get_path(config['paths']['checkpoint']['accuracy'].format(args.arch)),
                                   save_best_only=True,
                                   monitor='val_acc', mode='max')
    mcp_save_loss = ModelCheckpoint(utils.get_path(config['paths']['checkpoint']['loss'].format(args.arch)),
                                    save_best_only=True,
                                    monitor='val_loss', mode='min')

    
    step_size_train = np.ceil(train_generator.n / train_generator.batch_size)
    step_size_valid = np.ceil(valid_generator.n / valid_generator.batch_size)

    # Define how many iterations are required to complete a learning rate cycle
    stepSize = args.step * step_size_train

    clr = CyclicLR(mode=args.clr, 
                    base_lr=1e-4, 
                    max_lr=1e-2, 
                    step_size=stepSize)

    es = EarlyStopping(monitor='val_loss', 
                        patience=EPOCHS//2, 
                        mode='min', 
                        #restore_best_weights=True, 
                        min_delta=0.005,
                        verbose=1)

    history = model.fit(train_preprocessed,
                                  epochs=EPOCHS,
                                  verbose=1,
                                  #steps_per_epoch=step_size_train,
                                  validation_data=validation_preprocessed,
                                  #validation_steps=step_size_valid,
                                  callbacks=[es, clr, mcp_save_acc, mcp_save_loss],
                                  #workers=64,
                                  #use_multiprocessing=False,
                                  #max_queue_size=32
                                  )

    os.makedirs(utils.get_path(config['paths']['plot']['base'].format(args.arch)), exist_ok=True)    


    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(utils.get_path(config['paths']['plot']['accuracy'].format(args.arch, args.batch, args.step, args.opt, args.clr)))

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(utils.get_path(config['paths']['plot']['loss'].format(args.arch, args.batch, args.step, args.opt, args.clr)))



    accuracy = np.max(history.history['val_acc'])
    loss = np.min(history.history['val_loss'])

    print('Best accuracy model: {}'.format(accuracy))
    print('Best loss model: {}'.format(loss))


    # Save results
    with open('results.txt', 'a') as f:
        f.write('accuracy\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.arch, args.batch, args.step, args.opt, args.clr, accuracy))
        f.write('loss\t{}\t{}\t{}\t{}\t{}\t{}\n\n'.format(args.arch, args.batch, args.step, args.opt, args.clr, loss))        

    
