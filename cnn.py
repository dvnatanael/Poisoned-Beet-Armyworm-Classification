import os

import json
import logging
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

from typing import Sequence, Union, Optional, Callable
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import \
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, \
    RandomFlip, RandomRotation
from keras.regularizers import l2

# user-defined
import mycallbacks
import util


class BeetArmywormDataset(util.ImageDataset):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

        normalize: Callable = partial(
            tf.data.Dataset.map,
            map_func=lambda img, label: (tf.cast(img, tf.float32) / 255, label),
            num_parallel_calls=tf.data.AUTOTUNE)

        # normalize the data
        self.train_set = normalize(self.train_set)
        self.valid_set = normalize(self.valid_set)
        self.test_set = normalize(self.test_set)

        self.train_set.class_names = self.train_set._input_dataset.class_names
        self.valid_set.class_names = self.valid_set._input_dataset.class_names
        self.test_set.class_names = self.test_set._input_dataset.class_names

        self.train_set.file_paths = self.train_set._input_dataset.file_paths
        self.valid_set.file_paths = self.valid_set._input_dataset.class_names
        self.test_set.file_paths = self.test_set._input_dataset.class_names


class CNN(Sequential):
    def __init__(self, num_classes: int, name: str = None):
        # Define the layers
        preprocessor = Sequential(
            [RandomFlip(),
             # uncommenting this will cause tf to use a while_loop for a lot of functions
             # RandomContrast(factor=0.02),
             RandomRotation(factor=0.5)
             ],
            name='preprocessor')
        convolution_layer = Sequential(
            [Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization(),
             MaxPooling2D(pool_size=(2, 2)),
             Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization(),
             MaxPooling2D(pool_size=(2, 2)),
             Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization(),
             MaxPooling2D(pool_size=(2, 2)),
             Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization(),
             MaxPooling2D(pool_size=(2, 2)),
             Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization(),
             MaxPooling2D(pool_size=(2, 2)),
             Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                    padding='same', use_bias=False),
             BatchNormalization()
             ],
            name='convolution_layer')
        classifier = Sequential(
            [Flatten(),
             Dropout(rate=0.50),
             Dense(256, activation='elu', kernel_regularizer=l2(1e-3)),
             Dropout(rate=0.25),
             Dense(32, activation='elu', kernel_regularizer=l2(1e-3)),
             Dropout(rate=0.25),
             Dense(32, activation='elu', kernel_regularizer=l2(1e-3)),
             Dense(8, activation='elu', kernel_regularizer=l2(1e-3)),
             Dense(8, activation='elu', kernel_regularizer=l2(1e-3)),
             # output layer, use softmax to convert network output to probabilities
             Dense(num_classes, activation='softmax')
             ],
            name='classifier')
        layers = [preprocessor, convolution_layer, classifier]
        super().__init__(layers, name)


def create_model(
        input_shape: Sequence,
        num_classes: int,
        *,
        name: Optional[str] = None,
        optimizer: Optional[Union[str | keras.optimizers.Optimizer]] = None,
        loss: Optional[str | keras.losses.Loss] = None) -> CNN:
    """Create the model."""
    model = CNN(num_classes, name)
    model.build(input_shape)
    model.compile(optimizer, loss, metrics=['accuracy', 'Precision', 'Recall'])
    logger.info(util.get_model_summary(
        model,
        line_length=config.get('logger', {}).get('line_length'),
        expand_nested=True))
    logger.info(f'{loss=} {optimizer=}')

    return model


def train(
        model: keras.Model,
        data: util.ImageDataset,
        callbacks: dict[str, str],
        *,
        save_path: str = None,
        load_existing: bool = False,
        **kwargs) -> keras.callbacks.History:
    save_path = save_path or './model'
    # create model subdirectory to store model checkpoints
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        logging.info(f'Created directory {save_path}.')
    if load_existing:
        model.load_weights(os.path.join(save_path, f'{model.name}.h5'))

    # set up the callbacks
    callbacks = [
        mycallbacks.ReduceLROnPlateau(**callbacks.get('reduce lr on plateau', {})),
        mycallbacks.EarlyStopping(**callbacks.get('early stop', {})),
        mycallbacks.RecordWriter()
    ]

    return model.fit(
        data.train_set,
        validation_data=data.valid_set,
        callbacks=callbacks,
        **kwargs)


def main(configs: list):
    loader_config, model_config, optimizers = configs

    # load the data
    data = BeetArmywormDataset('./data/beet_armyworm', **loader_config)

    # update the configuration
    in_shape, out_shape = map(lambda x: x.shape, data.train_set.element_spec)
    __ = model_config.setdefault('model', {}).update({
        'input_shape': in_shape,
        'num_classes': out_shape[-1]})

    # set up the optimizer
    optimizer_name: str = model_config['model'].get('optimizer', 'RMSprop')
    optimizer: keras.optimizers.Optimizer = getattr(keras.optimizers, optimizer_name)
    model_config['model']['optimizer'] = optimizer(**optimizers.get(optimizer_name, {}))

    # create the model
    model = create_model(**model_config['model'])

    # train the model
    history = train(
        model,
        data,
        model_config.get('callbacks'),
        **model_config['fit'])
    util.save_model(model, './model')
    util.save_history(history, './model', model_name=model.name)

    # evaluate the model accuracy
    scores = model.evaluate(data.test_set, verbose=1, return_dict=True)
    s_yaml = util.YamlString(typ='safe')
    s_yaml.default_flow_style = False
    logger.info(s_yaml.dump({'scores': scores}))

    label_dict = dict(enumerate(data.class_names))
    logger.info(s_yaml.dump({'labels dictionary': label_dict}))
    # confusion matrix
    confusion_matrix = util.confusion_matrix(model, data.test_set)
    logger.info(f'confusion_matrix:\n{confusion_matrix}')

    # model predictions
    # todo plot prediction probabilities and their corresponding images
    s_yaml.default_flow_style = True
    pred_proba = model.predict(data.test_set)
    predictions = pd.Series(np.argmax(pred_proba, axis=-1))
    print(s_yaml.dump({
        'prediction probabilities': pred_proba.tolist(),
        'predictions': predictions.tolist()}))


logger = util.setup_logging(__name__, add_stream_handler=True)
if __name__ == '__main__':
    logger.info(f'Running {os.path.split(__file__)[-1]}')
    logger.info(f'Using GPU {gpu}' if (gpu := tf.test.gpu_device_name()) else f'Using CPU')

    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with open('config.yaml', 'r') as config_file:
        config, *model_configs = list(yaml.load_all(config_file))

    main(model_configs)
    logger.info(f'Finished running {os.path.split(__file__)[-1]}')
