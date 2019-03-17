import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock

class MaskTrackNet(nn.Module):

    def __init__(self):
        
        super(MaskTrackNet, self).__init__()

        # Resnet 34
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # BGR + Mask

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_conv_layer(block, 64, layers[0])
        self.layer2 = self._make_conv_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_conv_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_conv_layer(block, 512, layers[3], stride=2)

        self.deconv1 = self._make_deconv_layer(512, 256)
        self.deconv2 = self._make_deconv_layer(256, 128)
        self.deconv3 = self._make_deconv_layer(128, 64)
        self.deconv4 = self._make_deconv_layer(64, 1, stride=4)

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
    def _make_deconv_layer(in_planes, out_planes, stride=2):
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

    @staticmethod
    def _align_shape(x, desired_shape):
        left = int((x.shape[2] - desired_shape[2]) / 2)
        top = int((x.shape[3] - desired_shape[3]) / 2)
        x = x[:,:,left:left+desired_shape[2],top:top+desired_shape[3]]
        return x

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
        x = self._align_shape(x, pool2.shape)
        # print(x.shape)
        x = x + pool2

        x = self.deconv2(x)
        x = self._align_shape(x, pool1.shape)
        # print(x.shape)
        x = x + pool1

        x = self.deconv3(x)
        x = self._align_shape(x, pool0.shape)
        # print(x.shape)
        x = x + pool0

        x = self.deconv4(x)
        x = self._align_shape(x, input_shape)
        # print(x.shape)

        return x

    def load_pretrained_model(self, pretrained_model):
        own_state = self.state_dict()
        for name, param in pretrained_model.items():
            if name in own_state:
                if name is 'conv1.weight':
                    own_state[name][:,1:,:,:].copy_(param.data)
                else:
                    own_state[name].copy_(param.data)
                # own_state[name].requires_grad = False


if __name__ == '__main__':

    mt_net = MaskTrackNet()

    input = Variable(torch.randn(1, 4, 200, 200))
    output = mt_net(input)
    print(output.shape)