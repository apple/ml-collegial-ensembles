import os
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.logger import logger
from pathlib import Path
import contextlib
import time
import zipfile
import pickle


def retry(func):
    def inner(*args, **kwargs):
        max_trials = 100
        counter = 0
        while True:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                logger.warning('Failed to fetch data: {}'.format(str(e)))

                if counter >= max_trials:
                    raise ValueError('Max trials reached.')

                counter += 1
                time.sleep(30)
                logger.warning('Retrying [{}/{}]'.format(counter, max_trials))
    return inner


@contextlib.contextmanager
def local_numpy_seed(seed):
    """
    Within the context, the numpy seed will be fixed to the specified `seed`.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class BaseData:
    def __init__(self):
        self.datase_name = None
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.num_train_samples = None
        self.num_val_samples = None
        self.input_shape = None
        self.current_epoch = 0.

    def _load_data(self):
        raise NotImplementedError()

    def _print(self):
        logger.info('Dataset name: {}'.format(self.dataset_name))
        logger.info('Number of training samples: {:,}'.format(self.num_train_samples))
        logger.info('Number of validation samples: {:,}'.format(self.num_val_samples))

    def _subsample_train(self, num_samples, deterministic):
        if isinstance(num_samples, int):
            num_samples = min(num_samples, self.num_train_samples)
            with local_numpy_seed(seed=17) if deterministic else contextlib.suppress():
                indexes = np.random.choice(range(self.num_train_samples), num_samples, replace=False)
                self.x_train = self.x_train[indexes]
                if self.y_train is not None:
                    self.y_train = self.y_train[indexes]
            self.num_train_samples = num_samples

    def _get_iterator(self, batch_size, is_val=False, allow_smaller_batch=False):
        _num_samples = self.num_train_samples if not is_val else self.num_val_samples
        _X = self.x_train if not is_val else self.x_val
        _Y = self.y_train if not is_val else self.y_val

        indexes = np.random.permutation(range(_num_samples))
        for i in range(0, _num_samples, batch_size):
            if not allow_smaller_batch and (i + batch_size) > _num_samples:
                break
            if not is_val:
                self.current_epoch += batch_size / _num_samples
            yield _X[indexes[i: i + batch_size]], _Y[indexes[i: i + batch_size]]

    def next_batch(self, batch_size):
        raise NotImplementedError()

    def get_fixed_batch(self, batch_size, is_val=False):
        _X = self.x_train if not is_val else self.x_val
        _Y = self.y_train if not is_val else self.y_val
        indexes = np.arange(len(_Y))
        return _X[indexes[:batch_size]], _Y[indexes[:batch_size]]


class DownSampledImageNet(BaseData):
    """
    Chrabaszcz, Patryk, Ilya Loshchilov, and Frank Hutter. "A downsampled variant of imagenet as
    an alternative to the cifar datasets." arXiv preprint arXiv:1707.08819 (2017).
    """
    def __init__(self, train_path, val_path, resolution, subsample_train=None,
                 subsample_deterministic=True, layout='NCHW', demean=True):
        """
        :param train_path: path to Imagenet{resolution}_train.zip file or files.
        :param val_path: path to Imagenet{resolution}_val.zip file.
        :param resolution: resolution of the dataset, should be in {16, 32, 64}.
        :param subsample_train: how many samples to subsample from the training data. `None` means
            no subsampling.
        :param subsample_deterministic: if `True`, the subsampling will be determnistic in the
            following sense: 1. Calling twice this object with the same `subsample_train` value
            will yield the same data. 2. Calling twice this object with different `subsample_train`
            will yield the same first `n` samples where `n` is the minimum between the two values of
            `subsample_train`.
        """
        super().__init__()
        assert resolution in [16, 32, 64], "resolutions supported are {16, 32, 64} but {} was fed".format(resolution)

        if not isinstance(train_path, list):
            train_path = [train_path]

        self.train_path = train_path
        self.val_path = val_path

        self.resolution = resolution

        self.dataset_name = 'ImageNet-{0}x{0}'.format(self.resolution)

        (self.x_train, self.y_train), (self.x_val, self.y_val) = self._load_data()

        self.num_train_samples = self.x_train.shape[0]
        self.num_val_samples = self.x_val.shape[0]

        super()._subsample_train(subsample_train, subsample_deterministic)

        self._print()

    @retry
    def _load_data(self):
        def unzip(path, dest):
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(dest)

        def load(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)

            # dirty hack because mean image is not stored in validation data
            if 'mean' in d:
                self.mean = d['mean']

            x = d['data']
            x = x.astype(np.float32) / 255.

            x = x.reshape((-1, 3, self.resolution, self.resolution))
            y = np.array(d['labels']) - 1 # shift to 0 - 999
            return x, y.astype(np.int32)

        data_root = Path(__file__).resolve().parents[0]

        base_dest = os.path.join(data_root, 'ds-imagenet-{0}x{0}'.format(self.resolution))
        train_dest = os.path.join(base_dest, 'train')
        val_dest = os.path.join(base_dest, 'val')

        for p in self.train_path:
            unzip(p, train_dest)

        unzip(self.val_path, val_dest)

        # Loading everything in memory (~12GB for 64x64 ImageNet)
        # To reduce memory footprint, one can add some logic to load the sharded dataset
        # one file at a time. Samples in the files were randomly sampled without replacement
        # from the full dataset. See details in the original paper.
        train_pathes = [os.path.join(train_dest, f) for f in os.listdir(train_dest)
                                if 'train' in f]
        val_pathes = [os.path.join(val_dest, f) for f in os.listdir(val_dest)
                                if 'val' in f]

        xy_train = [load(p) for p in train_pathes]
        x_train = np.concatenate([x for x, _ in xy_train])
        y_train = np.concatenate([y for _, y in xy_train])

        xy_val = [load(p) for p in val_pathes]
        x_val = np.concatenate([x for x, _ in xy_val])
        y_val = np.concatenate([y for _, y in xy_val])

        return (x_train, y_train), (x_val, y_val)

    def next_batch(self, batch_size, is_val=False, allow_smaller_batch=False):
        return self._get_iterator(batch_size, is_val=is_val, allow_smaller_batch=allow_smaller_batch)



class TorchDSImageNetWrapper(Dataset):
    """
    Wraps BaseDownSampledImageNet in a Torch dataset.
    """
    def __init__(self, x, y, transform=None, mean=None):
        self.x = x
        self.y = y
        resolution = x.size(2)
        self.transform = transform
        if mean is not None:
            self.mean = torch.Tensor(mean).view((3, resolution, resolution))
        else:
            self.mean = mean

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        if self.mean is not None:
            x = x - self.mean

        return x, y

    def __len__(self):
        return self.x.size(0)

