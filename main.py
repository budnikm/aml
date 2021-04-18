import os
import shutil
import time
import math
import pickle
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.models as models

from lib.networks.mobilenet_v2 import MobileNetV2

from lib.networks.imageretrievalnet import init_network, extract_vectors
from lib.layers.loss import ContrastiveLoss, TripletLoss, ContrastiveDistLoss, CrossEntropyLoss, CrossEntropyDistLoss, MultiSimilarityLoss, RKdAD

from lib.datasets.traindataset import TuplesDataset, TuplesDatasetTS, TuplesDatasetTSWithSelf, RegressionTS, RegressionTSOnlyPos, TuplesDatasetRand
from lib.datasets.traindataset import RandomTriplet, RandomTripletAsym, TuplesDatasetTSRand

from lib.datasets.testdataset import configdataset
from lib.datasets.datahelpers import collate_tuples, collate_tuples_dist, cid2filename
from lib.utils.download import download_train, download_test
from lib.utils.whiten import whitenlearn, whitenapply
from lib.utils.evaluate import compute_map_and_print
from lib.utils.general import get_data_root, htime
from lib import cli, parse_args

from training import train, validate

import torch.nn.functional as F

training_dataset_names = ['retrieval-SfM-120k']
test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('mobilenet_v3')
model_names.append('efficientnet_b3')
model_names.append('efficientnet_b3_new') # which one?

pool_names = ['mac', 'spoc', 'gem', 'gemmp']

loss_names = ['contrastive', 'triplet', 'contrastive_dist', 'cross_entropy', 'cross_entropy_dist', 'multi', 'rkd']
mode_names = ['ts', 'ts_self', 'reg',  'reg_only_pos', 'std', 'rand', 'rand_tpl', 'rand_tpl_a', 'ts_rand']

teacher_names = ['vgg16', 'resnet101']
optimizer_names = ['sgd', 'adam']

min_loss = float('inf')

def main():
    global min_loss

    # manually check if there are unknown test datasets
    for dataset in args.test_datasets.split(','):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    data_root = '/nfs/nas4/mbudnik/dataset_descs/data/datasets'
    download_train(data_root)
    download_test(data_root)

    directory = parse_args.from_args_to_string(args)

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    log_out = args.directory+'/log.txt'
    log = open(log_out,'a')
    loss_log_out = args.directory+'/loss_log.txt'
    loss_log = open(loss_log_out,'a')
    
    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # set random seeds
    # TODO: maybe pass as argument in future implementation?
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # initialize model
    if args.pretrained:
        print(">> Using pre-trained model '{}'".format(args.arch))
    else:
        print(">> Using model from scratch (random weights) '{}'".format(args.arch))

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['pooling'] = args.pool
    model_params['local_whitening'] = args.local_whitening
    model_params['regional'] = args.regional
    model_params['whitening'] = args.whitening
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = args.pretrained
    model_params['teacher'] = args.teacher
    model = init_network(model_params)

    # move network to gpu
    model.cuda()
    if args.teacher == 'resnet101':
        args.feat_path = '/nfs/nas4/mbudnik/dataset_descs/data/features/retrieval-sfm-120k_retrievalSfM120k-resnet101-gem.npy'
        args.feat_val_path = '/nfs/nas4/mbudnik/dataset_descs/data/features/retrieval-SfM-30k_retrievalSfM120k-resnet101-gem.npy'
    else:
        args.feat_path = '/nfs/nas4/mbudnik/dataset_descs/data/features/retrieval-sfm-120k_retrievalSfM120k-vgg16-gem.npy'
        args.feat_val_path = '/nfs/nas4/mbudnik/dataset_descs/data/features/retrieval-SfM-30k_retrievalSfM120k-vgg16-gem.npy'
    
    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'contrastive_dist':
        criterion = ContrastiveDistLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'cross_entropy':
        criterion = CrossEntropyLoss(temp=args.temp).cuda()
    elif args.loss == 'cross_entropy_dist':
        criterion = CrossEntropyDistLoss(temp=args.temp).cuda()
    elif args.loss == 'multi':
        criterion = MultiSimilarityLoss().cuda()
    elif args.loss == 'rkd':
        criterion = RKdAD().cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    # parameters split into features, pool, whitening 
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    parameters = []
    # add feature parameters
    parameters.append({'params': model.features.parameters()})
    # add local whitening if exists
    if model.lwhiten is not None:
        parameters.append({'params': model.lwhiten.parameters()})
    # add pooling parameters (or regional whitening which is part of the pooling layer!)
    if not args.regional:
        # global, only pooling parameter p weight decay should be 0
        if args.pool == 'gem':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
    else:
        # regional, pooling parameter p weight decay should be 0, 
        # and we want to add regional whitening if it is there
        if args.pool == 'gem':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
        if model.pool.whiten is not None:
            parameters.append({'params': model.pool.whiten.parameters()})
    # add final whitening if exists
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})

    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    # TODO: maybe pass as argument in future implementation?
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> Finding the last checkpoint")
            all_file = os.listdir(args.directory)
            last_ckpt = 0
            ckpt_iter = 0
            for f in all_file:
                if f.startswith('model_epoch'):
                    ckpt_temp = int(all_file[ckpt_iter].split('.')[0].split('model_epoch')[1])
                    if ckpt_temp > last_ckpt:
                        last_ckpt = ckpt_temp
                ckpt_iter += 1
            resume_last = os.path.join(args.directory, 'model_epoch'+str(last_ckpt)+'.pth.tar')
            if os.path.isfile(resume_last):
                print(">> Loading checkpoint:\n>> '{}'".format(resume_last))
                checkpoint = torch.load(resume_last)
                start_epoch = checkpoint['epoch']
                min_loss = checkpoint['min_loss']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                        .format(resume_last, checkpoint['epoch']))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
            else:
                print(">> No checkpoint found at '{}'".format(resume_last))

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.mode == 'ts':
        tr_dataset = TuplesDatasetTS
    elif args.mode == 'ts_self':
        tr_dataset = TuplesDatasetTSWithSelf
    elif args.mode == 'ts_rand':
        tr_dataset = TuplesDatasetTSRand  
    elif args.mode == 'rand':
        tr_dataset = TuplesDatasetRand
    elif args.mode == 'rand_tpl':
        tr_dataset = RandomTriplet
    elif args.mode == 'rand_tpl_a':
        tr_dataset = RandomTripletAsym
    elif args.mode == 'reg' or args.mode == 'reg_only_pos':
        tr_dataset = RegressionTS
    else:
        tr_dataset = TuplesDataset

    train_dataset = tr_dataset(
            name=args.training_dataset,
            mode='train',
            imsize=args.image_size,
            nnum=args.neg_num,
            qsize=args.query_size,
            poolsize=args.pool_size,
            feat_path=args.feat_path,
            transform=transform,
            nexamples=args.nexamples
        )
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, sampler=None,
            drop_last=True, collate_fn=collate_tuples
        )
        
    #----------------------- VALIDATION -----------------------------------
    if args.val:
        if args.mode in ['std', 'rand_tpl']:
            vl_dataset = TuplesDataset
        elif args.mode == 'rand':
            vl_dataset = TuplesDatasetRand
        elif args.mode == 'ts_rand':
            vl_dataset = TuplesDatasetTSRand
        else:
            vl_dataset = TuplesDatasetTS
        
        val_dataset = vl_dataset(name=args.training_dataset, mode='val',
                imsize=args.image_size, nnum=args.neg_num, qsize=float('Inf'),
                poolsize=float('Inf'), feat_path=args.feat_val_path, transform=transform)
        
        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
                drop_last=True, collate_fn=collate_tuples
                )


    loss_log.write("epoch, train_loss, val_loss\n")
    for epoch in range(start_epoch, args.epochs):
        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()        
        
        # train for one epoch on train set
        
        loss = train(train_loader, model, criterion, optimizer, epoch, log, args)
        
        loss_log.write('%s, %s' %(epoch, loss))
        # evaluate on validation set
        if args.val and (epoch + 1) % args.val_freq == 0:
            with torch.no_grad():
                loss = validate(val_loader, model, criterion, epoch, args)
                loss_log.write(', %s' % loss)
                
        loss_log.write('\n')

        # remember best loss and save checkpoint
        is_best = False
        if args.val and (epoch + 1) % args.val_freq == 0:
            is_best = loss < min_loss
            min_loss = min(loss, min_loss)
        elif args.val == False:
            is_best = loss < min_loss
            min_loss = min(loss, min_loss)
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'meta': model.meta,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.directory)
        if is_best:
            save_checkpoint_best({
                'epoch': epoch + 1,
                'meta': model.meta,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),
            }, args.directory)
    log.close()
    loss_log.close()


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def save_checkpoint_best(state, directory):
    filename_best = os.path.join(directory, 'model_best.pth.tar')
    torch.save(state, filename_best)


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    main()