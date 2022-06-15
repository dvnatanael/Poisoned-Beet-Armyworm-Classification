import time
import itertools

from keras import backend, callbacks
from keras.callbacks import Callback

import util

logger = util.setup_logging(__name__)
record: list = []


class ReduceLROnPlateau(callbacks.ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        global record
        old_lr = backend.get_value(self.model.optimizer.learning_rate)
        super().on_epoch_end(epoch, logs)
        new_lr = backend.get_value(self.model.optimizer.learning_rate)
        if new_lr != old_lr:
            str_lr = f'{new_lr:.6f}' if new_lr >= 1e-6 else new_lr
            record.append(
                f'Model has not improved for {self.patience} epochs. '
                f'Reducing learning rate to {str_lr}.')


class EarlyStopping(callbacks.EarlyStopping):
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            global record
            record.append(
                f'Model has not improved for {self.patience} epochs.'
                f' Stopping training.')


class RecordWriter(Callback):
    start: float
    epoch: int = 0

    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        global record
        record = []

        logger.info('Training started.')
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        logger.debug(f'Epoch: {self.epoch}/{self.params["epochs"]}')
        self.start = time.perf_counter()
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        global record
        time_elapsed = int(time.perf_counter() - self.start)

        msg = ' - '.join([
            f'{time_elapsed}s',
            *(f'{k}: {v}' for k, v in logs.items())])
        for msg in itertools.chain(record, [msg]):
            logger.debug(msg)
        record = []

        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        global record
        for msg in record:
            logger.debug(record)
        record = []
        logger.info(
            f'Training finished. Model was trained for {self.epoch} epochs.')

        super().on_train_end(logs)
