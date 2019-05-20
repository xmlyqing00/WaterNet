import os
import argparse
import sys
import time
import configparser
import torch
from torch.utils import model_zoo

from src.AANet import FeatureNet, DeconvNet
from src.dataset import WaterDataset_RGB
from src.avg_meter import AverageMeter
from src.osvos_layers import class_balanced_cross_entropy_loss

def adjust_learning_rate(optimizer, start_lr, epoch, online_mode):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""
    if online_mode:
        decay_iters = 10
    else:
        decay_iters = 40
    lr = start_lr * (0.1 ** (epoch // decay_iters))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_AANet():

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch AANet Training')
    parser.add_argument(
        '--online', action='store_true',
        help='If the online flag is set, model will be trained in online mode, user must provide video name.')
    parser.add_argument(
        '--total-epochs', default=int(cfg['params_osvos']['total_epochs']), type=int, metavar='N',
        help='Number of total epochs to run (default 100).')
    parser.add_argument(
        '--lr', default=float(cfg['params_osvos']['lr']), type=float, metavar='LR', 
        help='Initial learning rate.')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-v', '--video-name', default=None, type=str, metavar='VIDEO_NAME',
        help='Test video name (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    assert(not args.online or (args.online and args.checkpoint))
    assert(not args.online or (args.online and args.video_name))

    if args.online and args.lr == float(cfg['params_osvos']['lr']):
        args.lr /= 10

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': int(cfg['params_osvos']['num_workers']),
            'pin_memory': bool(cfg['params_osvos']['pin_memory'])
        }

    if not args.online:
        dataset = WaterDataset_RGB(
            mode='train_offline',
            dataset_path=cfg['paths']['dataset'],
            input_size=(int(cfg['params_osvos']['input_w']), int(cfg['params_osvos']['input_h']))
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(cfg['params_osvos']['batch_size']),
            shuffle=True,
            **dataset_args
        )
    else:
        dataset = WaterDataset_RGB(
            mode='train_online',
            dataset_path=cfg['paths']['dataset'],
            input_size=(int(cfg['params_osvos']['input_w']), int(cfg['params_osvos']['input_h'])),
            test_case=args.video_name
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(cfg['params_osvos']['batch_size']),
            shuffle=False,
            **dataset_args
        )

    # Model
    feature_net = FeatureNet.to(device)
    deconv_net = DeconvNet().to(device)

    #Optimizor
    optimizer = torch.optim.SGD(
        params=OSVOS_net.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    # Load pretrained model
    start_epoch = 0
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print('Load checkpoint \'{}\''.format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            OSVOS_net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print('No checkpoint found at \'{}\''.format(args.checkpoint))
    else:
        print('Load pretrained VGG 19bn.')
        vgg19bn_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
        pretrained_model = model_zoo.load_url(vgg19bn_url)
        OSVOS_net.load_pretrained_model(pretrained_model)

    # Set train mode
    OSVOS_net.to(device).train()

    epoch_endtime = time.time()
    if not os.path.exists(cfg['paths']['models']):
        os.mkdir(cfg['paths']['models'])

    epoch_time = AverageMeter()

    training_mode = 'Offline'
    if args.online:
        training_mode = 'Online'

    for epoch in range(start_epoch, args.total_epochs):
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        batch_endtime = time.time()
        running_loss_tr = []
        for j in range(5):
            running_loss_tr.append(AverageMeter())

        if args.online:
            adjust_learning_rate(optimizer, args.lr, epoch - start_epoch, args.online)   
        else:
            adjust_learning_rate(optimizer, args.lr, epoch, args.online)   
        lr = -1
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        print('\n=== {0} Training Epoch: [{1:4}/{2:4}]\tlr: {3:.8f} ==='.format(
            training_mode, epoch, args.total_epochs - 1, lr
        ))

        for i, sample in enumerate(train_loader):

            img, label = sample['img'].to(device), sample['label'].to(device)
            outputs = OSVOS_net(img)

            layer_losses = [0] * len(outputs)
            for j in range(0, len(outputs)):
                layer_losses[j] = class_balanced_cross_entropy_loss(outputs[j], label, size_average=False)
                running_loss_tr[j].update(layer_losses[j].item())
            loss = (1 - epoch / args.total_epochs) * sum(layer_losses[:-1]) + layer_losses[-1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if args.online or ((i + 1) % 10 == 0 or (i + 1) == len(train_loader)):

                batch_time.update(time.time() - batch_endtime)
                batch_endtime = time.time()

                print('Batch: [{0:4}/{1:4}]\t'
                      'Time: {batch_time.val:.0f}s ({batch_time.sum:.0f}s)  \t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                      i, len(train_loader) - 1, 
                      batch_time=batch_time, loss=losses))
                
                # for j in range(5):
                #     print('\tRunning loss {0}: {loss.val:.4f} ({loss.avg:.4f})'.format(
                #         j, loss=running_loss_tr[j]))

        epoch_time.update(time.time() - epoch_endtime)
        epoch_endtime = time.time()

        print('Time: {epoch_time.val:.0f}s ({epoch_time.sum:.0f}s)  \t'
              'Avg loss: {loss.avg:.4f}'.format(
              epoch_time=epoch_time, loss=losses))

        if (epoch + 1) % 10 == 0 or (i + 1) == args.total_epochs:
            suffix = ''
            if args.online:
                suffix = '_' + args.video_name
            model_path = os.path.join(cfg['paths']['models'], 'cp_AANet_{0}{1}.pth.tar'.format(epoch, suffix))
            torch.save(
                obj={
                    'epoch': epoch,
                    'model': OSVOS_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': losses.avg,
                },
                f=model_path
            )
            
            print('Model saved.')

if __name__ == '__main__':
    train_AANet()
