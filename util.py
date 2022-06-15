import os
import sys
import time

import json
import logging
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

from typing import Optional, Callable
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# todo allow users to pass custom formatters and handlers
def setup_logging(
        logger_name: str,
        log_level: Optional = logging.DEBUG,
        *,
        add_stream_handler=False) -> logging.Logger:
    """Set up the logger."""
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '{asctime}|[{levelname:^8s}]|{name}|{message}', style='{')
    formatter.default_time_format = '%Y%m%d %H.%M.%S'
    formatter.default_msec_format = '%s.%03d'

    debug_handler = logging.FileHandler(f'debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    handler = logging.FileHandler(f'info.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if add_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.NOTSET)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class ImageDataset:
    def __init__(self, directory, **kwargs):
        from keras.utils import image_dataset_from_directory
        _ = kwargs.setdefault('validation_split', 0.2)
        load: Callable = partial(image_dataset_from_directory, **kwargs)

        dataset = tf.data.Dataset
        self.train_set: dataset = load(f'{directory}/train', subset='training')
        self.valid_set: dataset = load(f'{directory}/train', subset='validation')
        self.test_set: dataset = load(f'{directory}/test', subset=None, validation_split=None)
        self.class_names = self.train_set.class_names

        num_train_data = len(self.train_set.file_paths)
        num_valid_data = len(self.valid_set.file_paths)
        num_test_data = len(self.test_set.file_paths)
        logger.info(f'Found {num_train_data + num_valid_data + num_test_data} '
                    f'files belonging to {len(self.train_set.class_names)} classes.')
        logger.info(f'Using {num_train_data} files for training.')
        logger.info(f'Using {num_valid_data} files for validation.')
        logger.info(f'Using {num_test_data} files for testing.')


class YamlString(YAML):
    def dump(self, data, **kwargs) -> str:
        stream = StringIO()
        super().dump(data, stream, **kwargs)
        return stream.getvalue()


def get_model_summary(
        *models: tf.keras.Model,
        line_length: Optional[int] = 79,
        expand_nested: bool = False) -> str:
    if line_length < 1:
        logger.warning(f'Invalid line length (got {line_length}). '
                       f' Using line length of 79 characters.')
    elif line_length is None:
        line_length = 79

    summary = []
    for model in models:
        model.summary(
            line_length or 79,
            print_fn=lambda s: summary.append(s),
            expand_nested=expand_nested)
    return '\n'.join(summary)


def save_model(model: tf.keras.Model, save_path: str = None) -> None:
    filename = os.path.join(save_path or './model', f'{model.name}')

    # save model architecture as json
    with open(f'{filename}.json', 'w') as f:
        f.write(json.dumps(model.to_json(), indent=4))
        logger.info('Model architecture saved successfully.')

    # save model weights as h5
    model.save_weights(f'{filename}.h5')
    logger.info('Model weights saved successfully.')


def save_history(
        history: tf.keras.callbacks.Callback,
        save_path: str = None,
        *,
        model_name: str = None) -> None:
    filename = os.path.join(save_path or './model', model_name or 'model')

    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    yaml.explicit_start = True
    # save model train history as yaml
    with open(f'{filename}_train_history.yaml', 'a') as f:
        now = f'{time.strftime("%Y%m%d %H.%M.%S")}.{(time.time_ns() // 1000) % 1000}'
        train_history = {k: list(map(float, v)) for k, v in history.history.items()}
        doc = {'timestamp': now, 'train history': train_history}
        yaml.dump(doc, f)

    # save train history as png
    figure: plt.Figure = plot_train_history(history)
    figure.savefig(f'{filename}_train_history.png', bbox_inches='tight', transparent=True)


def plot_train_history(history: tf.keras.callbacks.History) -> plt.Figure:
    """Plot the model training hisotry."""

    def plotter(metric: str):
        ax.plot(history.history[metric], label=f'train')
        try:
            ax.plot(history.history[f'val_{metric}'], label=f'validation')
        except KeyError:
            pass
        ax.set_ylabel(metric.title())
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')

    rows = cols = int(np.ceil(len(history.history)))
    fig: plt.Figure = plt.figure(figsize=(12, 8))
    gs: plt.GridSpec = fig.add_gridspec(rows, cols)
    for i, key in enumerate([k for k in history.history if 'val' not in k]):
        ax: plt.Axes = fig.add_subplot(gs[divmod(i, cols)])
        plotter(key)

    fig.suptitle('Train History')
    fig.tight_layout()
    return fig


def confusion_matrix(model: tf.keras.Model, data: tf.data.Dataset) -> np.ndarray:
    confusion: tf.Tensor = tf.math.confusion_matrix([], [], len(data.class_names))
    for images, labels in data:
        confusion += tf.math.confusion_matrix(
            labels=np.argmax(labels, axis=-1),
            predictions=np.argmax(model.predict(images), axis=-1))

    # convert `confusion` into a `pd.DataFrame`
    confusion: pd.DataFrame = pd.DataFrame(confusion)
    confusion.index.name = 'label'
    confusion.columns.name = 'predict'
    return confusion


logger = setup_logging(__name__)
