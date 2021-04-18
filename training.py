import time
import torch


def train(train_loader, model, criterion, optimizer, epoch, log, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        if args.mode == 'rand':
            outputs = torch.zeros(nq, target[0].shape[0]).cuda()
        for q in range(nq):
            ni = len(input[q])
            output = torch.zeros(model.meta['outputdim'], ni).cuda()
            if args.mode in ['ts', 'ts_self', 'ts_rand', 'reg', 'reg_only_pos', 'rand_tpl_a']: 
                if args.sym == True:
                    for imi in range(ni):
                        output[:, imi] = model(input[q][imi].cuda()).squeeze()
                else:
                    for imi in range(ni):
                        if imi == 0:
                            output[:, imi] = model(input[q][imi].cuda()).squeeze()
                        else:
                            output[:, imi] = torch.tensor(input[q][imi]).float().cuda()
                            
            elif args.mode in ['std', 'rand_tpl']:
                for imi in range(ni):
                    output[:, imi] = model(input[q][imi].cuda()).squeeze()
            else:
                for imi in range(ni):
                    output[:, imi] = model(input[q][imi].cuda()).squeeze()
                outputs[q,:] = output.squeeze()
            if args.mode != 'rand':
                loss = criterion(output, target[q].t().cuda())
                losses.update(loss.item())
                loss.backward()
        if args.mode == 'rand':
            targets = torch.stack(target).cuda()
            loss = criterion(outputs, targets)
            losses.update(loss.item())
            loss.backward()
        
        if (i + 1) % args.update_every == 0:
            # do one step for multiple batches
            # accumulated gradients are used
            optimizer.step()
            # zero out gradients so we can 
            # accumulate new ones over batches
            optimizer.zero_grad()
            # print('>> Train: [{0}][{1}/{2}]\t'
            #       'Weight update performed'.format(
            #        epoch+1, i+1, len(train_loader)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            out = '>> Train: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t \
                   Data {data_time.val:.3f} ({data_time.avg:.3f})\t \
                   Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)
            print(out)
            log.write(out+'\n')

    return losses.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        output = torch.zeros(model.meta['outputdim'], nq*ni).cuda()
        if args.mode == 'rand':
            outputs = torch.zeros(nq, target[0].shape[0]).cuda()
        for q in range(nq):
            if args.mode in ['ts', 'reg', 'reg_only_pos', 'ts_self', 'ts_rand', 'rand_tpl_a']:
                if args.sym == True:
                    for imi in range(ni):
                        output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()
                else:
                    for imi in range(ni):
                        if imi == 0:
                            output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()
                        else:
                            output[:, q*ni + imi] = torch.tensor(input[q][imi]).float().cuda()
            elif args.mode == 'rand':
                for imi in range(ni):
                    output[:, imi] = model(input[q][imi].cuda()).squeeze()
            else:
                for imi in range(ni):
                    # compute output vector for image imi of query q
                    output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()
                
        # no need to reduce memory consumption (no backward pass):
        # compute loss for the full batch
        
        if args.mode == 'rand':
            targets = torch.stack(target).cuda().t()
            loss = criterion(output.t(), targets.t())
        else:
            if args.sym:
                loss = criterion(output, torch.cat(target).cuda().t())
            else:
                loss = criterion(output, torch.cat(target).cuda())

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(val_loader):
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False
