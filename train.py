import os
import numpy as np
from utils.logger import logger
import argparse
import time

import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.resnext import CifarResNeXt, SmallImageNetResNeXt, ImageNetResNeXt
from data.datasets import retry, DownSampledImageNet, TorchDSImageNetWrapper


def config():
    """
    Configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True, choices=['cifar10', 'cifar100',
                                'imagenet16', 'imagenet32', 'imagenet64', 'imagenet'],
                                help= 'Which dataset to chose.')
    parser.add_argument('--train-path', type=str, nargs='+', default=None,
                                help='For imagenet16/32/64, path(s) to the train zip file(s). For '
                                'imagenet, path to training data. For cifar10 and cifar100, no '
                                'need to specify a path.')
    parser.add_argument('--val-path', type=str, default=None, help='see train-path')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-gpus', type=int, default=None, help='number of GPUs to use.')
    parser.add_argument('--num-workers', type=int, default=8, help='how many subprocesses to '
                                'use for data loading')
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--learning-rate', type=float, default=None, help='initial learning rate')
    parser.add_argument('--depth', type=int, default=None, help='depth of the ResNe(X)t')
    parser.add_argument('--cardinality', type=int, default=1, help='Number of pathes i.e. groups '
                                'in the group convolutions')
    parser.add_argument('--base-width', type=int, default=None, help='base width of the ResNeXt'
                                'network following the ResNeXt parametrization defined in '
                                'Xie, Saining, et al. "Aggregated residual transformations for '
                                'deep neural networks."')
    parser.add_argument('--logging-steps', type=int, default=50, help='log training progress '
                                'every few steps')
    params = parser.parse_args()

    if 'cifar' not in params.dataset:
        assert params.train_path is not None, "--train-path should be specified."
        assert params.val_path is not None, "--val-path should be specified."

    def update_config(default):
        for k, v in default.items():
            if getattr(params, k) is None:
                setattr(params, k, v)
        return params

    if params.dataset == 'imagenet':
    	params.train_path = params.train_path[0]
        default = {'epochs': 100, 'batch_size': 256, 'weight_decay': 0.0001, 'learning_rate': 0.1,
                'depth': 50, 'cardinality': 1, 'base_width': 64, 'depth': 50}
    elif 'imagenet' in params.dataset: # downsampled ImageNet
        default = {'epochs': 80, 'batch_size': 1024, 'weight_decay': 0.0005, 'learning_rate': 0.08,
                'cardinality': 1, 'base_width': 128, 'depth': 29}
    elif 'cifar' in params.dataset:
        default = {'epochs': 300, 'batch_size': 128, 'weight_decay': 0.0005, 'learning_rate': 0.1,
                'cardinality': 1, 'base_width': 128, 'depth': 29}

    params = update_config(default)
    return params


def get_accuracy(output, target, topk=(1, 5)):
    """
    Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


class ResNeXt:
    def __init__(self, *args, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'cifar' in self.dataset:

            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]

            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            val_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])

            dset_class = dset.CIFAR10 if self.dataset == 'cifar10' else dset.CIFAR100
            self.n_labels = 10 if self.dataset == 'cifar10' else 100

            train_data = retry(dset_class)('./', train=True, transform=train_transform, download=True)
            val_data = retry(dset_class)('./', train=False, transform=val_transform, download=True)

        elif self.dataset == 'imagenet':

            train_transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

            val_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

            train_data = dset.ImageFolder(self.train_path, train_transform)
            val_data = dset.ImageFolder(self.val_path, val_transform)

            self.n_labels = 1000

        elif 'imagenet' in self.dataset:

            self.resolution = int(self.dataset[8:])

            train_transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.resolution, padding=4), transforms.ToTensor()])
            val_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

            data = DownSampledImageNet(self.train_path, self.val_path, self.resolution)
            mean = data.mean / 255.
            train_data = TorchDSImageNetWrapper(torch.Tensor(data.x_train), data.y_train,
                            transform=train_transform, mean=mean)
            val_data = TorchDSImageNetWrapper(torch.Tensor(data.x_val) , data.y_val,
                            transform=val_transform, mean=mean)
            del data

            self.n_labels = 1000


        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def train(self):

        def _train():
            net.train()
            loss_avg = 0.0
            eps_avg = []

            start = time.time()
            for step, (data, target) in enumerate(self.train_loader):
                target = target.type(torch.LongTensor)
                if self.num_gpus:
                    data = data.cuda()
                    target = target.cuda()

                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

                # forward
                output = net(data)

                # backward
                optimizer.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                eps_avg.append(self.batch_size / (time.time() - start))
                start = time.time()

                # exponential moving average
                loss_avg = loss_avg * 0.2 + float(loss) * 0.8

                if (step % self.logging_steps) == 0:
                    logger.info('epoch: {}, step: {}, loss: {:.5f}, eps: {:.0f}' \
                                    .format(epoch, step, loss_avg, np.mean(eps_avg)))

                    eps_avg = []

        def _val():
            net.eval()

            if 'imagenet' in self.dataset:
                correct_top1 = 0
                correct_top5 = 0
            else:
                correct = 0

            for batch_idx, (data, target) in enumerate(self.val_loader):
                target = target.type(torch.LongTensor)
                if self.num_gpus:
                    data = data.cuda()
                    target = target.cuda()
                data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

                # forward
                output = net(data)

                if 'imagenet' in self.dataset:
                    top1, top5 = get_accuracy(output, target, topk=(1, 5))
                    correct_top1 += top1
                    correct_top5 += top5
                else:
                    pred = output.data.max(1)[1]
                    correct += float(pred.eq(target.data).sum())

            if 'imagenet' in self.dataset:
                error = {'top1': 1. - float(correct_top1) / len(self.val_loader.dataset),
                            'top5': 1. - float(correct_top5) / len(self.val_loader.dataset)}
            else:
                error = 1. - correct / len(self.val_loader.dataset)
            return error


        logger.info('Dataset size is {:,}.'.format(len(self.train_loader.dataset)))
        logger.info('{:,} steps per epoch.'.format(int(len(self.train_loader.dataset) / self.batch_size)))

        # create network
        if 'cifar' in self.dataset:
            net = CifarResNeXt(self.cardinality, self.depth, self.n_labels, self.base_width)
        elif self.dataset == 'imagenet':
            net = ImageNetResNeXt(self.cardinality, self.depth, self.n_labels, self.base_width)
        elif 'imagenet' in self.dataset:
            net = SmallImageNetResNeXt(self.cardinality, self.depth, self.n_labels, self.base_width,
                                       self.resolution)

        print(net)

        available_gpus = torch.cuda.device_count()
        self.num_gpus = self.num_gpus if self.num_gpus else 0
        if available_gpus > 0 and self.num_gpus > 0:
            net = torch.nn.DataParallel(net, device_ids=list(range(self.num_gpus)))
            net.cuda()
        logger.info('{} GPUs detected, using {} GPUs.'.format(available_gpus, self.num_gpus))

        optimizer = torch.optim.SGD(net.parameters(), self.learning_rate, momentum=0.9,
                                    weight_decay=self.weight_decay, nesterov=False)

        # *********
        # Main loop
        # *********

        learning_rate = self.learning_rate
        for epoch in range(self.epochs):

            # learning rate schedule
            if 'cifar' in self.dataset:
                if epoch in [150, 225]:
                    learning_rate = learning_rate * 0.1
            elif self.dataset == 'imagenet':
                learning_rate = self.learning_rate * (0.1 ** (epoch // 30))
            elif 'imagenet' in self.dataset:
                if epoch in [20, 40, 60]:
                    learning_rate /= 5.

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


            _train() # train for one epoch

            logger.info('*' * 15)
            logger.info('Starting validation.')
            val_error = _val()

            if 'imagenet' in self.dataset:
                logger.info('[Validation] top1-error: {:.3f}, top5-eror: {:.3f}'.format(
                                                val_error['top1'], val_error['top5']))
            else:
                logger.info('[Validation] error: {:.3f}'.format(val_error))
            logger.info('*' * 15)


if __name__ == '__main__':
    config = config()
    model = ResNeXt(**vars(config))
    model.train()
