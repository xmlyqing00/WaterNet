import os
import argparse
import sys
import time
import cv2
import torch
import configparser
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.network import RGBMaskNet
from src.dataset import WaterDataset_RGBMask
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays
from src.utils import load_image_in_PIL, iou_tensor


def eval_RGBMaskNet():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    if sys.platform == 'darwin':
        cfg_dataset = 'dataset_mac'
    elif sys.platform == 'linux':
        cfg_dataset = 'dataset_ubuntu'

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch RGBMaskNet Testing')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-v', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
    parser.add_argument(
        '-m', '--model-name', default='RGBMaskNet', type=str,
        help='Model name for the ouput segmentation, it will create a subfolder under the out_folder.')
    parser.add_argument(
        '-o', '--out-folder', default=os.path.join(cfg['paths'][cfg_dataset], 'results/'), type=str, metavar='PATH',
        help='Folder for the output segmentations.')
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help='Evaluate the video with groundtruth.')
    parser.add_argument(
        '--sample', action='store_true',
        help='The video sequence has been sampled.')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')

    water_thres = 0.5

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': int(cfg['params_RGBM']['num_workers']),
            'pin_memory': bool(cfg['params_RGBM']['pin_memory'])
        }

    dataset = WaterDataset_RGBMask(
        mode='eval',
        dataset_path=cfg['paths'][cfg_dataset], 
        test_case=args.video_name,
        eval_size=(int(cfg['params_RGBM']['eval_w']), int(cfg['params_RGBM']['eval_h']))
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

    # Model
    rgbmask_net = RGBMaskNet().to(device)

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        rgbmask_net.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Set ouput path
    out_path = os.path.join(args.out_folder, args.model_name + '_segs', args.video_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if args.sample:
        out_full_path = out_path + '_full'
        if not os.path.exists(out_full_path):
            os.makedirs(out_full_path)

    # Start testing
    rgbmask_net.eval()
    running_time = AverageMeter()
    running_endtime = time.time()
    
    # First frame annotation
    pre_frame_mask = dataset.get_first_frame_label()
    eval_size = pre_frame_mask.shape[-2:]
    first_frame_seg = TF.to_pil_image(pre_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    if args.sample:
        first_frame_seg.save(os.path.join(out_full_path, '0.png'))
    pre_frame_mask = pre_frame_mask.unsqueeze(0).to(device)

    if args.benchmark:
        gt_folder = os.path.join(cfg['paths'][cfg_dataset], 'test_annots', args.video_name)
        gt_list = os.listdir(gt_folder)
        gt_list.sort(key = lambda x: (len(x), x))
        gt_list.pop(0)
    avg_iou = 0

    with torch.no_grad():
        for i, sample in enumerate(tqdm(eval_loader)):

            img = sample['img'].to(device)     
            img_mask = torch.cat([img, pre_frame_mask], 1)  

            output = rgbmask_net(img_mask)

            pre_frame_mask = output.detach()
            # seg_raw = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())
            # seg_raw.save(os.path.join(out_path, 'raw_%d.png' % (i + 1)))

            zero_tensor = torch.zeros(pre_frame_mask.shape).to(device)
            one_tensor = torch.ones(pre_frame_mask.shape).to(device)
            pre_frame_mask = torch.where(pre_frame_mask > water_thres, one_tensor, zero_tensor)
            seg = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())

            if args.sample:
                seg.save(os.path.join(out_full_path, f'{i + 1}.png'))

                if i + 1 in [1, 50, 100, 150, 199]:
                    seg.save(os.path.join(out_path, f'{i + 1}.png'))        
            
            else:
                seg.save(os.path.join(out_path, f'{i + 1}.png'))        

            running_time.update(time.time() - running_endtime)
            running_endtime = time.time()

            # if args.benchmark:
            #     gt_seg = load_image_in_PIL(os.path.join(gt_folder, gt_list[i])).convert('L')
            #     gt_tf = TF.to_tensor(gt_seg).to(device).type(torch.int)

            #     print(pre_frame_mask.squeeze(0).type(torch.int).max())
            #     print(gt_tf.max())
            #     iou = iou_tensor(pre_frame_mask.squeeze(0).type(torch.int), gt_tf)
            #     avg_iou += iou.item()
            #     print('iou:', iou.item())

            # print('Segment: [{0:4}/{1:4}]\t'
            #     'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\t'.format(
            #     i + 1, len(eval_loader), running_time=running_time))
   

    # if args.benchmark:
    #     print('total_iou:', avg_iou)
    #     avg_iou /= len(eval_loader)
    #     print('avg_iou:', avg_iou, 'frame_num:', len(eval_loader))

    if args.sample:
        mask_folder = args.video_name + '_full'
    else:
        mask_folder = args.video_name
    run_cvt_images_to_overlays(args.video_name, mask_folder, cfg['paths'][cfg_dataset], args.model_name, eval_size)

    
if __name__ == '__main__':
    eval_RGBMaskNet()
