# Collegial Ensembles

This software project accompanies the research paper, [Collegial Ensembles](https://arxiv.org/abs/2006.07678).


Collegial Ensembles can be used to analytically derive optimal group convolution modules without having to train a single model.
- Primal optimum: Optimal model with the same budget of parameters.
- Dual optimum: Equivalent model with minimal number of parameters.


## Reproducing Results on CIFAR

To reproduce the baselines on CIFAR-10:

- `python train.py -d cifar10 --cardinality 1 --base-width 128 --num-gpus 8`

- `python train.py -d cifar10 --cardinality 3 --base-width 64 --num-gpus 8`

To reproduce the primal optimum:

- `python train.py -d cifar10 --cardinality 37 --base-width 10 --num-gpus 8`

And the dual optimum:

- `python train.py -d cifar10 --cardinality 10 --base-width 10 --num-gpus 8`

Replace `cifar10` by `cifar100` to obtain results for CIFAR-100.


## Reproducing Results on Downsampled ImageNet

To reproduce the baselines on ImageNet32x32:

- `python train.py -d imagenet32 --train-path TRAIN_PATH --val-path VAL_PATH --cardinality 1 --base-width 128 --num-gpus 8`

- `python train.py -d imagenet32 --train-path TRAIN_PATH --val-path VAL_PATH --cardinality 3 --base-width 64 --num-gpus 8`

where `TRAIN_PATH` and `VAL_PATH` are the paths to the downsampled ImageNet zip files available at http://image-net.org/download-images.

To reproduce the primal optimum:

- `python train.py -d imagenet32 --train-path TRAIN_PATH --val-path VAL_PATH --cardinality 37 --base-width 10 --num-gpus 8`

And the dual optimum:

- `python train.py -d imagenet32 --train-path TRAIN_PATH --val-path VAL_PATH --cardinality 10 --base-width 10 --num-gpus 8`

Replace `imagenet32` by `imagenet64` to obtain results for ImageNet64x64

## Reproducing Results on ImageNet


To reproduce the ResNet-50 baselines on ImageNet:

- `python train.py -d imagenet --train-path TRAIN_PATH --val-path VAL_PATH --depth 50 --cardinality 1 --base-width 64 --num-gpus 8`

- `python train.py -d imagenet --train-path TRAIN_PATH --val-path VAL_PATH --depth 50 --cardinality 32 --base-width 4 --num-gpus 8`

To reproduce the primal optimum:

- `python train.py -d imagenet --train-path TRAIN_PATH --val-path VAL_PATH --depth 50 --cardinality 12 --base-width 10 --num-gpus 8`

And the dual results:

- `python train.py -d imagenet --train-path TRAIN_PATH --val-path VAL_PATH --depth 50 --cardinality 4 --base-width 16 --num-gpus 8`

To get results for ResNeXt-101, replace `--depth 50` with `--depth 101`.


## Dependencies

- python 3.6
- pytorch 1.5.0
- torchvision 0.6.0

## Credit

The implementation of ResNeXt architecture is based on:

- https://github.com/facebookresearch/ResNeXt
- https://github.com/prlz77/ResNeXt.pytorch
