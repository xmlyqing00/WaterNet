import os
import argparse
import sys
import time
import configparser
import torch
from torch.utils import model_zoo
from torchvision import transforms

from src.network import MaskTrackNet
from src.dataset import WaterDataset
from src.avg_meter import AverageMeter


def adjust_learning_rate(optimizer, start_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_MaskTrackNet():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('paths.conf')

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch FCNResNet Training')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N',
        help='Manual epoch number (useful on restarts, default 0).')
    parser.add_argument(
        '--total-epochs', default=int(cfg['params']['total_epoch']), type=int, metavar='N',
        help='Number of total epochs to run (default 100).')
    parser.add_argument(
        '--lr', '--learning-rate', default=float(cfg['params']['lr']), type=float,
        metavar='LR', help='Initial learning rate.')
    parser.add_argument(
        '--resume', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': 4,
            'pin_memory': True
        }

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    dataset = WaterDataset(
        mode='train',
        dataset_path=cfg['path']['training_dataset_path'],
        input_transforms=transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ]),
        target_transforms=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        **dataset_args
    )

    # Model
    mt_net = MaskTrackNet().to(device)

    # Criterion and Optimizor
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.SGD(
        params=mt_net.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    # Load pretrained model
    if args.resume:
        if os.path.isfile(args.resume):
            print('Load checkpoint \'{}\''.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            mt_net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('No checkpoint found at \'{}\''.format(args.resume))
    else:
        print('Load pretrained ResNet 34.')
        # resnet32_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        print(mt_net)
        pretrained_model = torch.load(os.path.join(cfg['path']['models_path'], 'resnet34-333f7ec4.pth'))
        print(pretrained_model)
        mt_net.load_pretrained_model(pretrained_model)

    # Start training
    mt_net.train()
    epoch_endtime = time.time()
    if not os.path.exists(cfg['path']['models_path']):
        os.mkdir(cfg['path']['models_path'])

    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.total_epochs):
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        batch_endtime = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)   

        for i, (input, target) in enumerate(train_loader):
            
            input, target = input.to(device), target.to(device)

            output = mt_net(input)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if i % 100 == 0:

                batch_time.update(time.time() - batch_endtime)
                batch_endtime = time.time()

                print('Epoch: [{0}/{1} | {2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, args.total_epochs, i, len(train_loader),
                      batch_time=batch_time, loss=losses))

        epoch_time.update(time.time() - epoch_endtime)
        epoch_endtime = time.time()

        torch.save(
            obj={
                'epoch': epoch,
                'model': mt_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': losses.avg,
            },
            f=os.path.join(cfg['path']['models_path'], 'checkpoint_{0}.pth.tar'.format(epoch))
        )

        print('Epoch: [{0}/{1}]\t'
              'Time {epoch_time.val:.3f} ({epoch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
              epoch, args.total_epochs, 
              epoch_time=epoch_time, loss=losses))
        
        print('Model saved.')


if __name__ == '__main__':
    train_MaskTrackNet()
