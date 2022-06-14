import os
import time

import pickle
import logging
from ruamel.yaml import YAML

from typing import Sequence, Union, Optional, Callable, Iterable, Any
from functools import partial, reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.callbacks import Callback, EarlyStopping, History, ReduceLROnPlateau
from keras.layers import \
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, \
    RandomFlip, RandomRotation
from keras.regularizers import l2

# user-defined
import util


class BeetArmywormDataset(util.ImageDataset):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

        # assume that all class names in the train and test sets are the same
        self.class_names: list[str] = self.train_set.class_names

        # normalize the data
        normalize: Callable = partial(
            tf.data.Dataset.map,
            map_func=lambda img, label: (tf.cast(img, tf.float32) / 255, label),
            num_parallel_calls=tf.data.AUTOTUNE)
        self.train_set = normalize(self.train_set)
        self.valid_set = normalize(self.valid_set)
        self.test_set = normalize(self.test_set)


class CNN:
    model: keras.Model
    train_history: History
    _callbacks: list[Callback]
    save_path: str = './model'

    def __init__(
            self,
            input_shape: Sequence,
            num_classes: int,
            *,
            loss: Optional[str | keras.losses.Loss] = None,
            optimizer: Optional[Union[str | keras.optimizers.Optimizer]] = None,
            save_path: Optional[str] = None,
            model_name: Optional[str] = None,
            load_existing: bool = False,
            **kwargs):
        # create the model
        self.define_model(num_classes, name=model_name)
        self.model.build(input_shape)
        self.model.compile(optimizer, loss, metrics=['accuracy'])
        if load_existing:
            self.load_weights()
        logger.info(util.get_model_summary(
            self.model,
            line_length=kwargs.get('line_length'),
            expand_nested=True))
        logger.info(f'{loss=} {optimizer=}')

        self.callbacks = []

        # create model subdirectory to store model checkpoints
        self.save_path: str = save_path or self.save_path
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
            logging.info(f'Created directory {self.save_path}.')

    def define_model(self, num_classes: int, *args, name: Optional[str] = None) -> None:
        """Define the model."""
        preprocessor = Sequential(
            [
                RandomFlip(),
                RandomRotation(factor=0.5),
                # uncommenting this will cause tf to use a while_loop for a lot of functions
                # RandomContrast(factor=0.02)
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
             BatchNormalization()],
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
             Dense(num_classes, activation='softmax')],
            name='classifier')
        self.model = Sequential(
            [preprocessor, convolution_layer, classifier],
            name=name or 'model')

    def train(self, dataset: util.ImageDataset, **kwargs) -> History:
        """Train the model."""
        self.train_history: History = self.model.fit(
            dataset.train_set,
            validation_data=dataset.valid_set,
            callbacks=self.callbacks,
            **kwargs)
        return self.train_history

    @property
    def callbacks(self) -> list[Callback]:
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value: Union[list[Callback] | Callback]) -> None:
        if isinstance(value, Callback):
            self._callbacks = [value]
        elif isinstance(value, Iterable) \
                and reduce(lambda a, b: a and isinstance(b, Callback), value, True):
            self._callbacks = value
        else:
            raise ValueError(
                'Value must be of type `keras.callbacks.Callback` or '
                '`Iterable[keras.callbacks.Callback]`.')

    @callbacks.deleter
    def callbacks(self) -> None:
        self._callbacks = []

    def predict(self, x: Any):
        return self.model.predict(x)

    def load_weights(self, *, weights_path: Optional[str] = None) -> None:
        if weights_path is None:
            weights_path = os.path.join(self.save_path, f'{self.model.name}.h5')
        try:
            self.model.load_weights(weights_path)
        except FileNotFoundError:
            logger.warning('Model not found. Training a new model.')
        else:
            logger.info(f'Loaded weights from {weights_path}.')

    # todo modify
    def plot_train_history(self) -> plt.Figure:
        """Plot the model training hisotry."""
        history = self.train_history.history

        def plotter(ax: plt.Axes, train: str, val: str):
            ax.plot(history[train], label=f'train')
            ax.plot(history[val], label=f'validation')
            ax.set_title('Train History')
            ax.set_ylabel(train.title())
            ax.set_xlabel('Epoch')
            ax.legend(loc='upper left')

        fig: plt.Figure = plt.figure(figsize=(12, 8))
        gs: plt.GridSpec = fig.add_gridspec(1, 2)
        ax: plt.Axes = fig.add_subplot(gs[0, 0])
        plotter(ax, 'accuracy', 'val_accuracy')  # plot accuracy
        ax: plt.Axes = fig.add_subplot(gs[0, 1])
        plotter(ax, 'loss', 'val_loss')  # plot loss

        fig.tight_layout()
        return fig

    def save(self, *, plot_train_history: bool = False) -> None:
        # check whether the model has been trained
        if 'train_history' not in self.__dir__():
            msg = 'This model has not been trained. Train the model first by calling `train()`.'
            logger.critical(msg)
            raise RuntimeError(msg)

        filename = os.path.join(self.save_path, f'{self.model.name}')

        # save model architecture as json
        with open(f'{filename}.json', 'w') as f:
            f.write(self.model.to_json())
        logger.info('Model architecture saved successfully.')

        # save model weights as h5
        self.model.save_weights(f'{filename}.h5')
        logger.info('Model weights saved successfully.')

        # todo allow the user to choose to save the training history as pickle or as yaml
        # save model train history as pkl
        with open(f'{filename}_train_history.pkl', 'wb') as f:
            pickle.dump(self.train_history.history, f)
        # save model train history as yaml
        with open(f'{filename}_train_history.yaml', 'a') as f:
            now = f'{time.strftime("%Y%m%d %H.%M.%S")}.{(time.time_ns() // 1000) % 1000}'
            doc = {'timestamp': now, 'train history': self.train_history.history}
            yaml.explicit_start = True
            yaml.dump(doc, f)
            yaml.explicit_start = False
        logger.info('Train history saved successfully.')

        # save train history as png
        if plot_train_history:
            figure: plt.Figure = self.plot_train_history()
            figure.savefig(f'{filename}_train_history.png', bbox_inches='tight', transparent=True)


def main(config: dict):
    # load the data
    data = BeetArmywormDataset('./data/beet_armyworm', **config['imageloader'])

    # update the configuration
    in_shape, out_shape = map(lambda x: x.shape, data.train_set.element_spec)
    __ = config.setdefault('model', {}).update({
        'input_shape': in_shape,
        'num_classes': out_shape[-1]})

    # create the model
    model = CNN(**config['model'], line_length=config.get('logger', {}).get('line_length'))
    # todo setup the callbacks
    # my_callback: "CustomCallback" = ...
    # reduce_lr_on_plateau: ReduceLROnPlateau = ...
    # record_writer: "RecordWriter" = ...
    # early_stop: EarlyStopping = ...
    # model.callbacks = [my_callback, reduce_lr_on_plateau, record_writer, early_stop]
    # train the model
    model.train(data, **config['model']['fit'])
    model.save(plot_train_history=True)

    # use the models for prediction
    predictions = pd.Series(np.argmax(model.predict(data), axis=-1))
    with pd.option_context('display.max_rows', 10):
        print(f'predictions:\n{predictions}', )


logger = util.setup_logging(__name__, add_stream_handler=True)
if __name__ == '__main__':
    __, filename = os.path.split(__file__)
    logger.info(f'Running {filename}')

    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with open('config.yaml', 'r') as f:
        config, optimizers = yaml.load_all(f)
    optimizer_name = config['model'].get('optimizer', 'RMSprop')
    config['model']['optimizer']: keras.optimizers.Optimizer \
        = getattr(keras.optimizers, optimizer_name)(**optimizers[optimizer_name])

    main(config)
