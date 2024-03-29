#!/usr/bin/env python

"""learningratefinder.py: Learning rate finder algorithm."""
__author__      = "David Bertoldi"
__email__       = "d.bertoldi@campus.unimib.it"


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile
import argparse
import utils
from processing import Processing
import models




class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
                       "Iterator", "Sequence"]

        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss

        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR, epochs=None,
             stepsPerEpoch=None, batchSize=32, sampleSize=2048,
             verbose=1, use_multiprocessing=True):
        # reset our class-specific variables
        self.reset()

        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch

        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
        self.on_batch_end(batch, logs))

        # check to see if we are using a data iterator

        
        self.model.fit(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])
            

        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # plot the learning rate vs. loss
        plt.plot(lrs, losses, color='green')
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Flower Recognition Neural Network')

    parser.add_argument('--batch', type=int, const=64, default=64, nargs='?', help='Batch size used during training')
    parser.add_argument('--arch', type=str, const='resnet18', default='resnet18', nargs='?', choices=models.ARCHITECTURES.keys(), help='Architecture')
    parser.add_argument('--opt', type=str, const='Adam', default='SGD', nargs='?', choices=models.OPTIMIZERS.keys(), help='Optimizer')
    parser.add_argument('--dropout', type=float, const=0.5, default=0.5, nargs='?', help='Dropout rate')
    parser.add_argument('--config', type=str, const='config.yml', default='config.yml', nargs='?', help='Configuration file')
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

    # Get the model
    architecture = models.ARCHITECTURES[args.arch]
    architecture = architecture(args.dropout)
    preprocessor = architecture.preprocess()
    model = architecture.get_model()
    
    #model.summary(show_trainable=True)
    target_size = architecture.size


    # Get the optimizer
    optimizer = models.OPTIMIZERS[args.opt]['get']()()
    learning_rate = models.OPTIMIZERS[args.opt]['lr']

    
    p = Processing(target_size=target_size,
                    batch_size=args.batch,
                    config=config,
                    preprocessor=preprocessor)
    



    train_preprocessed, validation_preprocessed, test_preprocessed  = p.get_dataset()
    train_cardinality, validation_cardinality = train_preprocessed.n, validation_preprocessed.n

    # Define how many iterations are required to complete a learning rate cycle
    step_size_train = np.ceil(train_cardinality / args.batch)

    model.compile(loss=config['training']['loss'], optimizer=optimizer, metrics=['acc'])

    lrf = LearningRateFinder(model)
    lrf.find(
        train_preprocessed,
        1e-10, 1e-1,
        epochs=10,
        stepsPerEpoch=step_size_train,
        batchSize=args.batch,
        use_multiprocessing=False)

    # plot the loss for the various learning rates and save the
    # resulting plot to disk
    lrf.plot_loss()
    plt.show()
    #plt.savefig(utils.get_path(config['paths']['plot']['lr']))
