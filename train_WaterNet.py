import os
import argparse
import sys
import time
import configparser
import torch
from torch.utils import model_zoo

from src.WaterNet import FeatureNet, DeconvNet
from src.dataset import WaterDataset_RGB
from src.avg_meter import AverageMeter

def adjust_learning_rate(optimizer, start_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""

    decay_iters = 40
    lr = start_lr * (0.1 ** (epoch // decay_iters))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_WaterNet():

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')
    cfg_dataset = 'dataset_ubuntu'

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch WaterNet Training')
    parser.add_argument(
        '--total-epochs', default=int(cfg['params_water']['total_epochs']), type=int, metavar='N',
        help='Number of total epochs to run (default 200).')
    parser.add_argument(
        '--lr', default=float(cfg['params_water']['lr']), type=float, metavar='LR', 
        help='Initial learning rate.')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': int(cfg['params_water']['num_workers']),
            'pin_memory': bool(cfg['params_water']['pin_memory'])
        }

    dataset = WaterDataset_RGB(
        mode='train_offline',
        dataset_path=cfg['paths'][cfg_dataset],
        input_size=(int(cfg['params_water']['input_w']), int(cfg['params_water']['input_h']))
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(cfg['params_water']['batch_size']),
        shuffle=True,
        **dataset_args
    )

    # Model
    feature_net = FeatureNet().to(device)
    deconv_net = DeconvNet().to(device)

    #Optimizor
    feature_net_optimizer = torch.optim.SGD(
        params=feature_net.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    deconv_net_optimizer = torch.optim.SGD(
        params=deconv_net.parameters(),
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
            feature_net.load_state_dict(checkpoint['feature_net'])
            deconv_net.load_state_dict(checkpoint['deconv_net'])
            feature_net_optimizer.load_state_dict(checkpoint['feature_net_optimizer'])
            deconv_net_optimizer.load_state_dict(checkpoint['deconv_net_optimizer'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print('No checkpoint found at \'{}\''.format(args.checkpoint))
    else:
        print('Load pretrained ResNet 34.')
        resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        pretrained_model = model_zoo.load_url(resnet34_url)
        feature_net.load_pretrained_model(pretrained_model)

    # Set train mode
    feature_net.train()
    deconv_net.train()

    # Criterion
    criterion = torch.nn.BCELoss().to(device)

    epoch_endtime = time.time()
    if not os.path.exists(cfg['paths']['models']):
        os.mkdir(cfg['paths']['models'])

    epoch_time = AverageMeter()

    training_mode = 'Offline'

    for epoch in range(start_epoch, args.total_epochs):
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        batch_endtime = time.time()
        running_loss_tr = []
        for j in range(5):
            running_loss_tr.append(AverageMeter())

        adjust_learning_rate(feature_net_optimizer, args.lr, epoch)   
        adjust_learning_rate(deconv_net_optimizer, args.lr, epoch)
        lr = -1
        for param_group in feature_net_optimizer.param_groups:
            lr = param_group['lr']
            break

        print('\n=== {0} Training Epoch: [{1:4}/{2:4}]\tlr: {3:.8f} ==='.format(
            training_mode, epoch, args.total_epochs - 1, lr
        ))

        for i, sample in enumerate(train_loader):

            img, label = sample['img'].to(device), sample['label'].to(device)
            
            feature_map, f0, f1, f2 = feature_net(img)
            output = deconv_net(feature_map, f0, f1, f2, img.shape)

            loss = criterion(output, label)
            feature_net_optimizer.zero_grad()
            deconv_net_optimizer.zero_grad()
            loss.backward()
            feature_net_optimizer.step()
            deconv_net_optimizer.step()

            losses.update(loss.item())

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):

                batch_time.update(time.time() - batch_endtime)
                batch_endtime = time.time()

                print('Batch: [{0:4}/{1:4}]\t'
                      'Time: {batch_time.val:.0f}s ({batch_time.sum:.0f}s)  \t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                      i, len(train_loader) - 1, 
                      batch_time=batch_time, loss=losses))
                
        epoch_time.update(time.time() - epoch_endtime)
        epoch_endtime = time.time()

        print('Time: {epoch_time.val:.0f}s ({epoch_time.sum:.0f}s)  \t'
              'Avg loss: {loss.avg:.4f}'.format(
              epoch_time=epoch_time, loss=losses))

        if (epoch + 1) % 10 == 0 or (i + 1) == args.total_epochs:
            suffix = ''
            model_path = os.path.join(cfg['paths']['models'], 'cp_WaterNet_{0}{1}.pth.tar'.format(epoch, suffix))
            torch.save(
                obj={
                    'epoch': epoch,
                    'feature_net': feature_net.state_dict(),
                    'deconv_net': deconv_net.state_dict(),
                    'feature_net_optimizer': feature_net_optimizer.state_dict(),
                    'deconv_net_optimizer': deconv_net_optimizer.state_dict(),
                    'loss': losses.avg,
                },
                f=model_path
            )
            
            print('Model saved.')

if __name__ == '__main__':
    train_WaterNet()
