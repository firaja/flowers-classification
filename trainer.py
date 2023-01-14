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
from processing import Processing

np.random.seed(42)
tf.random.set_seed(42)
    


CLRS = ['triangular', 'triangular2', 'exp']



def parse_arguments():
    parser = argparse.ArgumentParser(description='Flower Recognition Neural Network')

    parser.add_argument('--batch', type=int, const=64, default=64, nargs='?', help='Batch size used during training')
    parser.add_argument('--arch', type=str, const='resnet18', default='resnet18', nargs='?', choices=models.ARCHITECTURES.keys(), help='Architecture')
    parser.add_argument('--opt', type=str, const='Adam', default='SGD', nargs='?', choices=models.OPTIMIZERS.keys(), help='Optimizer')
    parser.add_argument('--clr', type=str, const='triangular', default='triangular', nargs='?', choices=CLRS, help='Cyclical learning rate')
    parser.add_argument('--step', type=float, const=8, default=8, nargs='?', help='Step size')
    parser.add_argument('--dropout', type=float, const=0.5, default=0.5, nargs='?', help='Dropout rate')
    parser.add_argument('--config', type=str, const='config.yml', default='config.yml', nargs='?', help='Configuration file')
    parser.add_argument('--mp', default=False, action='store_true', help='Enable mixed precision operations (16bit-32bit)')
    parser.add_argument('--da', default=False, action='store_true', help='Enable Data Augmentation')
    parser.add_argument('--freeze', type=float, const=0, default=0, nargs='?', help='Frozen layers')
    parser.add_argument('--epoch', type=int, const=50, default=50, nargs='?', help='Set the number of epochs')

    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = utils.read_configuration(args.config)

    # Do not allocate all the memory during initialization
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Enable 16bit operations
    if args.mp:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Mixed precision enabled!')
        print('Operations dtype: {}'.format(policy.compute_dtype))
        print('Variables dtype: {}'.format(policy.variable_dtype))
    
    # Get the model
    architecture = models.ARCHITECTURES[args.arch]
    architecture = architecture(args.dropout)
    preprocessor = architecture.preprocess()
    model = architecture.get_model()
    
    model.summary(show_trainable=True)
    target_size = architecture.size


    # Get the optimizer
    optimizer = models.OPTIMIZERS[args.opt]['get']()()
    learning_rate = models.OPTIMIZERS[args.opt]['lr']

    # Download and preprocess the dataset with data augmentation
    p = None
    if args.da:
        p = Processing(target_size=target_size,
                        batch_size=args.batch,
                        config=config,
                        preprocessor=preprocessor)
    # or without data augmentation
    else:
        p = Processing(target_size=target_size,
                        batch_size=args.batch,
                        config=config,
                        brightness_delta=0,
                        zoom_delta=0, 
                        flip=False, 
                        rotation=0)



    train_preprocessed, validation_preprocessed, test_preprocessed  = p.get_dataset()
    train_cardinality, validation_cardinality = train_preprocessed.n, validation_preprocessed.n

    
    # Finalize the model
    model.compile(loss=config['training']['loss'], optimizer=optimizer, metrics=['acc'])


    # Checkpoints
    mcp_save_acc = ModelCheckpoint(utils.get_path(config['paths']['checkpoint']['accuracy'].format(args.arch)),
                                   save_best_only=True,
                                   monitor='val_acc', mode='max')
    mcp_save_loss = ModelCheckpoint(utils.get_path(config['paths']['checkpoint']['loss'].format(args.arch)),
                                    save_best_only=True,
                                    monitor='val_loss', mode='min')

    

    # Define how many iterations are required to complete a learning rate cycle
    step_size_train = np.ceil(train_cardinality / args.batch)
    step_size_valid = np.ceil(validation_cardinality / args.batch)
    stepSize = args.step * step_size_train

    # Define the Cyclic Learnin Rate
    clr = CyclicLR(mode=args.clr, 
                    base_lr=learning_rate[0], 
                    max_lr=learning_rate[1],
                    step_size=stepSize)

    # Define the Early Stopping strategy
    es = EarlyStopping(monitor='val_acc', 
                        patience=10, 
                        mode='min', 
                        #restore_best_weights=True, 
                        min_delta=0.005,
                        verbose=1)

    # Train
    history = model.fit(train_preprocessed,
                                  epochs=args.epoch,
                                  verbose=1,
                                  steps_per_epoch=step_size_train,
                                  validation_data=validation_preprocessed,
                                  validation_steps=step_size_valid,
                                  callbacks=[clr, mcp_save_acc, mcp_save_loss, es],
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
    plt.savefig(utils.get_path(config['paths']['plot']['accuracy'].format(args.arch, args.batch, args.step, args.opt, args.clr, args.da)))

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(utils.get_path(config['paths']['plot']['loss'].format(args.arch, args.batch, args.step, args.opt, args.clr, args.da)))



    accuracy = np.max(history.history['val_acc'])
    loss = np.min(history.history['val_loss'])

    print('Best accuracy model: {}'.format(accuracy))
    print('Best loss model: {}'.format(loss))


    # Save results
    with open('results.txt', 'a') as f:
        f.write('accuracy\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(args.arch, args.batch, args.step, args.opt, args.clr, accuracy))
        f.write('loss\t{}\t{}\t{}\t{}\t{}\t{}\n\n'.format(args.arch, args.batch, args.step, args.opt, args.clr, loss))        

    
