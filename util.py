import sys
import logging
from functools import partial
from typing import Optional, Callable


# todo allow users to pass custom formatters and handlers
def setup_logging(
        logger_name: Optional[str] = None,
        debug_level: Optional = logging.DEBUG,
        *,
        add_stream_handler=False) -> logging.Logger:
    """Set up the logger."""
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(debug_level)

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


def get_model_summary(
        *models: "keras.Model",
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


class ImageDataset:
    def __init__(self, directory, **kwargs):
        import tensorflow as tf
        from keras.utils import image_dataset_from_directory
        _ = kwargs.setdefault('validation_split', 0.2)
        load: Callable = partial(image_dataset_from_directory, **kwargs)

        Dataset = tf.data.Dataset
        self.train_set: Dataset = load(f'{directory}/train', subset='training')
        self.valid_set: Dataset = load(f'{directory}/train', subset='validation')
        self.test_set: Dataset = load(f'{directory}/test', subset=None, validation_split=None)

        num_train_data = len(self.train_set.file_paths)
        num_valid_data = len(self.valid_set.file_paths)
        num_test_data = len(self.test_set.file_paths)
        logger.info(f'Found {num_train_data + num_valid_data + num_test_data} '
                    f'files belonging to {len(self.train_set.class_names)} classes.')
        logger.info(f'Using {num_train_data} files for training.')
        logger.info(f'Using {num_valid_data} files for validation.')
        logger.info(f'Using {num_test_data} files for testing.')


logger = setup_logging()
