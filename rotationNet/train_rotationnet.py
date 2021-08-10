import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--case', default='2', type=str,
                    help='viewpoint setup case (1 or 2)')

best_prec1 = 0
vcand = np.load('vcand_case2.npy')
nview = 20

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def main():
    global args, best_prec1, nview, vcand
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.case == '1':
        vcand = np.load('vcand_case1.npy')
        nview = 12
    elif args.case == '3':
        vcand = np.load('vcand_case3.npy')
        nview = 160

    if args.batch_size % nview != 0:
        print('Error: batch size should be multiplication of the number of views,', nview)
        exit()

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    # Get number of classes from train directory
    num_classes = len([name for name in os.listdir(traindir)])
    print("num_classes = '{}'".format(num_classes))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = FineTuneModel(model, args.arch, (num_classes+1) * nview )

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            # model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # model = torch.nn.DataParallel(model).cuda()
            model.cuda()

    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    ##optimizer = torch.optim.SGD(model.parameters(), args.lr,
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
#            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    sorted_imgs = sorted(train_loader.dataset.imgs)
    train_nsamp = int( len(sorted_imgs) / nview )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
#            transforms.Scale(256),
#            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader.dataset.imgs = sorted(val_loader.dataset.imgs)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch)

        # random permutation
        inds = np.zeros( ( nview, train_nsamp ) ).astype('int')
        inds[ 0 ] = np.random.permutation(range(train_nsamp)) * nview
        for i in range(1,nview):
            inds[ i ] = inds[ 0 ] + i
        inds = inds.T.reshape( nview * train_nsamp )
        train_loader.dataset.imgs = [sorted_imgs[ i ] for i in inds]
        train_loader.dataset.samples = train_loader.dataset.imgs

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        fname='rotationnet_checkpoint.pth.tar'
        fname2='rotationnet_model_best.pth.tar'
        if nview == 12:
            fname='rotationnet_checkpoint_case1.pth.tar'
            fname2='rotationnet_model_best_case1.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,fname,fname2)

            

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        nsamp = int( input.size(0) / nview )

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input).cuda()
        target_ = torch.LongTensor( target.size(0) * nview )

        # compute output
        output = model(input_var)
        num_classes = int( output.size( 1 ) / nview ) - 1
        output = output.view( -1, num_classes + 1 )

        ###########################################
        # compute scores and decide target labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax( output )
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[ :, :-1 ] - torch.t( output_[ :, -1 ].repeat( 1, output_.size(1)-1 ).view( output_.size(1)-1, -1 ) )
        # reshape output matrix
        output_ = output_.view( -1, nview * nview, num_classes )
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose( 1, 2, 0 )
        # initialize target labels with "incorrect view label"
        for j in range(target_.size(0)):
            target_[ j ] = num_classes
        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros( ( vcand.shape[ 0 ], num_classes, nsamp ) )
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]
        # for each sample #n, determine the best pose that maximizes the score for the target class (see Eq.(2))
        for n in range( nsamp ):
            j_max = np.argmax( scores[ :, target[ n * nview ], n ] )
            # assign target labels
            for k in range(vcand.shape[1]):
                target_[ n * nview * nview + vcand[ j_max ][ k ] * nview + k ] = target[ n * nview ]
        ###########################################

        target_ = target_.cuda()
        target_var = torch.autograd.Variable(target_)

        # compute loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda()
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # log_softmax and reshape output
            num_classes = int( output.size( 1 ) / nview ) - 1
            output = output.view( -1, num_classes + 1 )
            output = torch.nn.functional.log_softmax( output )
            output = output[ :, :-1 ] - torch.t( output[ :, -1 ].repeat( 1, output.size(1)-1 ).view( output.size(1)-1, -1 ) )
            output = output.view( -1, nview * nview, num_classes )

            # measure accuracy and record loss
            prec1, prec5 = my_accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0)/nview)
            top5.update(prec5.item(), input.size(0)/nview)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='rotationnet_checkpoint.pth.tar', filename2='rotationnet_model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename2)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print ('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def my_accuracy(output_, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    target = target[0:-1:nview]
    batch_size = target.size(0)

    num_classes = output_.size(2)
    output_ = output_.cpu().numpy()
    output_ = output_.transpose( 1, 2, 0 )
    scores = np.zeros( ( vcand.shape[ 0 ], num_classes, batch_size ) )
    output = torch.zeros( ( batch_size, num_classes ) )
    # compute scores for all the candidate poses (see Eq.(6))
    for j in range(vcand.shape[0]):
        for k in range(vcand.shape[1]):
            scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range( batch_size ):
        j_max = int( np.argmax( scores[ :, :, n ] ) / scores.shape[ 1 ] )
        output[ n ] = torch.FloatTensor( scores[ j_max, :, n ] )
    output = output.cuda()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
