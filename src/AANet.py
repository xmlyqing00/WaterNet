import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock

from src.network import FCNBase


class ParentNet(FCNBase):

    def __init__(self):

        super(ParentNet, self).__init__()

        # ResNet 34
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        
        # Conv module
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_conv_layer(block, 64, layers[0])
        self.layer2 = self._make_conv_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_conv_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_conv_layer(block, 512, layers[3], stride=2)

        # # Deconv module
        # self.upsample1 = self._make_upsample_layer(512, 256)
        # self.upsample2 = self._make_upsample_layer(256, 128)
        # self.upsample3 = self._make_upsample_layer(128, 64)
        # self.upsample4 = self._make_upsample_layer(64, 64, stride=4)

        # # Output mask
        # self.blend1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn_blend = nn.BatchNorm2d(64)
        # self.blend2 = nn.Conv2d(64, class_n, kernel_size=3, padding=1)
        # self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_conv_layer(self, block, planes, blocks, stride=1):
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
    def _make_upsample_layer(in_planes, out_planes, stride=2):

        assert(stride == 2 or stride == 4)

        layers = []
        if stride == 2:
            layers.append(
                nn.ConvTranspose2d(in_planes, out_planes, padding=1, output_padding=1, kernel_size=3, stride=2)
            )
        elif stride == 4:
            layers.append(
                nn.ConvTranspose2d(in_planes, out_planes, padding=1, output_padding=1, kernel_size=5, stride=4)
            )
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):

        # input_shape = x.shape
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/2
        # pool0 = x

        x = self.layer1(x) 
        x = self.layer2(x) # 1/2
        # pool2 = x 

        x = self.layer3(x) # 1/2
        # pool3 = x

        x = self.layer4(x)
        # x = self.upsample1(x)
        
        # x = x + pool3
        # x = self.upsample2(x)

        # x = x + pool2
        # x = self.upsample3(x)

        # x = x + pool0
        # x = self.upsample4(x)
        
        # x = self.blend1(x)
        # x = self.bn_blend(x)
        # x = self.relu(x)

        # x = self.blend2(x)
        # x = self.sigmoid(x)

        return x

class FeatureMatching(FCNBase):

    def __init__(self):

        super(FeatureMatching, self).__init__()

    def forward(self, x):
        return x