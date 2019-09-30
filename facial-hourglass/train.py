import os
import chainer

import sys
sys.path.append('.')
sys.path.append('..')

from datasets.dataset import KeypointDataset
from arguments import parse_args
from model import HourGlassNet
from updater import KeypointUpdater


def main(args):
    args = parse_args()

    # Get Data
    print('[INFO] Preparing data..')
    img_paths = glob.glob(args.img_glob)
    pts_paths = glob.glob(args.pts_glob)

    assert len(img_paths)==len(pts_paths), "Number of images and points must be equal"

    # Split train and test
    split_idx = int(args.dataset_ratio * len*img_path)
    trainset = KeypointDataset(
        img_paths[:args.train_ratio],
        pts_paths[:args.train_ratio],
        args.pts_mode)
    testset = KeypointDataset(
        img_paths[args.train_ratio:],
        pts_paths[args.train_ratio:],datasetdataset
        args.pts_mode)

    print('[INFO] Train dataset Length:',len(train_dataset))
    print('[INFO] Test dataset Length:',len(test_dataset))

    # Prepare iterators
    batchsize = args.batchsize
    train_iter = chainer.iterators.SerialIterator(trainset, batchsize)
    test_iter = chainer.iterators.SerialIterator(testset, batchsize)

    # Prepare model
    model = HourGlassNet()

    # Choose optimizer
    optmizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Updater
    updater= training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu_id)

    # Setup a Trainer
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=args.out)
