import os
import argparse
import models
import processing
import yaml
    


CLRS = ['triangular', 'triangular2', 'exp']

def get_path(path):
    return os.path.abspath(path)

def read_configuration(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flower Recognition Neural Network')

    parser.add_argument('--batch', type=int, const=32, default=32, nargs='?', help='Batch size used during training')
    parser.add_argument('--arch', type=str, const='densenet121', default='densenet121', nargs='?', choices=models.ARCHITECTURES.keys(), help='Architecture')
    parser.add_argument('--opt', type=str, const='Adam', default='Adam', nargs='?', choices=models.OPTIMIZERS.keys(), help='Optimizer')
    parser.add_argument('--clr', type=str, const='triangular2', default='triangular2', nargs='?', choices=CLRS, help='Cyclical learning rate')
    parser.add_argument('--step', type=float, const=0.001, default=0.001, nargs='?', help='Step size')
    parser.add_argument('--dropout', type=float, const=0.5, default=0.5, nargs='?', help='Dropout rate')
    parser.add_argument('--config', type=str, const='config.yml', default='config.yml', nargs='?', help='Configuration file')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = read_configuration(args.config)


    architecture = models.ARCHITECTURES[args.arch]
    
    model = architecture['get'](args.dropout)()
    
    target_size = architecture['size']
    
    optimizer = models.OPTIMIZERS[args.opt]['get']()()

    train_generator = processing.train_data_generator().flow_from_directory(directory=get_path(config['paths']['train']),
                                                                            batch_size=args.batch,
                                                                            shuffle=True,
                                                                            target_size=(target_size, target_size),
                                                                            interpolation=config['training']['interpolation'],
                                                                            class_mode=config['training']['mode'])

    valid_generator = processing.test_data_generator().flow_from_directory(directory=get_path(config['paths']['valid']),
                                                                            batch_size=args.batch,
                                                                            shuffle=True,
                                                                            target_size=(target_size, target_size),
                                                                            class_mode=config['training']['mode'])

    test_generator = processing.test_data_generator().flow_from_directory(directory=get_path(config['paths']['test']),
                                                                            batch_size=args.batch,
                                                                            shuffle=False,
                                                                            target_size=(target_size, target_size),
                                                                            class_mode=config['training']['mode'])

    model.compile(loss=config['training']['loss'], optimizer=optimizer, metrics=['acc'])
