import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock
from torchvision.models import vgg, vgg19_bn

from src.osvos_layers import center_crop, interp_surgery


class FCNBase(nn.Module):

    def __init__(self):
        super(FCNBase, self).__init__()
    
    @staticmethod
    def align_shape(x, desired_shape):
        left = int((x.shape[2] - desired_shape[2]) / 2)
        top = int((x.shape[3] - desired_shape[3]) / 2)
        x = x[:,:,left:left+desired_shape[2],top:top+desired_shape[3]]
        return x
    
    def load_pretrained_model(self, pretrained_model):
        own_state = self.state_dict()
        for name, param in pretrained_model.items():
            if name in own_state:
                own_state[name].copy_(param.data)

    def forward(self, x):
        raise NotImplementedError
        

class OSVOSNet(FCNBase):

    def __init__(self):

        super(OSVOSNet, self).__init__()

        lay_list = [[64, 64],
                    ['M', 128, 128],
                    ['M', 256, 256, 256, 256],
                    ['M', 512, 512, 512, 512],
                    ['M', 512, 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]

        stages = nn.modules.ModuleList()
        side_prep = nn.modules.ModuleList()
        score_dsn = nn.modules.ModuleList()
        upscale = nn.modules.ModuleList()
        upscale_ = nn.modules.ModuleList()

        # Construct the network
        for i in range(0, len(lay_list)):
            # Make the layers of the stages
            stages.append(self.make_layers(lay_list[i], in_channels[i]))

            # Attention, side_prep and score_dsn start from layer 2
            if i > 0:
                # Make the layers of the preparation step
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))

                # Make the layers of the score_dsn step
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)


    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.stages[0](x)

        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))

        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def load_pretrained_model(self, pretrained_model):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        _vgg = vgg19_bn(pretrained=False)
        inds = self.find_conv_layers(_vgg)

        k = 0
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i])):
                if isinstance(self.stages[i][j], nn.Conv2d):
                    self.stages[i][j].weight = deepcopy(_vgg.features[inds[k]].weight)
                    self.stages[i][j].bias = deepcopy(_vgg.features[inds[k]].bias)
                    k += 1

    @staticmethod
    def make_layers(cfg, in_channels):
        layers = []
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    @staticmethod
    def find_conv_layers(_vgg):
        inds = []
        for i in range(len(_vgg.features)):
            if isinstance(_vgg.features[i], nn.Conv2d):
                inds.append(i)
        return inds


class RGBMaskNet(FCNBase):

    def __init__(self):
        
        super(RGBMaskNet, self).__init__()

        # Resnet 34
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # BGR + Mask

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_conv_layer(block, 64, layers[0])
        self.layer2 = self.make_conv_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_conv_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_conv_layer(block, 512, layers[3], stride=2)

        self.deconv1 = self.make_deconv_layer(512, 256)
        self.deconv2 = self.make_deconv_layer(256, 128)
        self.deconv3 = self.make_deconv_layer(128, 64)
        self.deconv4 = self.make_deconv_layer(64, 1, stride=4)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_conv_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def make_deconv_layer(in_planes, out_planes, stride=2):
        layers = []
        if stride == 4:
            layers.append(nn.ConvTranspose2d(in_planes, 
                                             in_planes, 
                                             padding=1,
                                             output_padding=1,
                                             kernel_size=3, 
                                             stride=2))    
        layers.append(nn.ConvTranspose2d(in_planes, 
                                         out_planes,
                                         padding=1,
                                         output_padding=1,
                                         kernel_size=3, 
                                         stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):

        input_shape = x.shape
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/2
        pool0 = x

        x = self.layer1(x) 
        x = self.layer2(x) # 1/2
        pool1 = x 

        x = self.layer3(x) # 1/2
        pool2 = x

        x = self.layer4(x)

        # print(pool0.shape, pool1.shape, pool2.shape, x.shape)

        x = self.deconv1(x)
        x = self.aligh_shape(x, pool2.shape)
        # print(x.shape)
        x = x + pool2

        x = self.deconv2(x)
        x = self.aligh_shape(x, pool1.shape)
        # print(x.shape)
        x = x + pool1

        x = self.deconv3(x)
        x = self.aligh_shape(x, pool0.shape)
        # print(x.shape)
        x = x + pool0

        x = self.deconv4(x)
        x = self.aligh_shape(x, input_shape)
        # print(x.shape)

        x = self.sigmoid(x)

        return x

    def load_pretrained_model(self, pretrained_model):
        own_state = self.state_dict()
        for name, param in pretrained_model.items():
            if name in own_state:
                if name == 'conv1.weight':
                    own_state[name][:,1:,:,:].copy_(param.data)
                else:
                    own_state[name].copy_(param.data)
                # own_state[name].requires_grad = False


if __name__ == '__main__':

    # mt_net = RGBMaskNet()

    # input = Variable(torch.randn(1, 4, 200, 200))
    # output = mt_net(input)
    # print(output.shape)

    pfd_net = PureFeatureDetectorNet()
    print(pfd_net)