from torch import nn
from torchvision.models.resnet import BasicBlock


class FeatureNet(nn.Module):

    def __init__(self):

        super(FeatureNet, self).__init__()

        # ResNet 34
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        
        # Conv module
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_conv_layer(block, 64, layers[0])
        self.layer2 = self.make_conv_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_conv_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_conv_layer(block, 512, layers[3], stride=2)

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

    def load_pretrained_model(self, pretrained_model):
        own_state = self.state_dict()
        for name, param in pretrained_model.items():
            if name in own_state:
                own_state[name].copy_(param.data)
                
    def forward(self, x):

        # input_shape = x.shape
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # 1/2
        f0 = x

        x = self.layer1(x) 
        x = self.layer2(x) # 1/2
        f1 = x 

        x = self.layer3(x) # 1/2
        f2 = x

        f3 = self.layer4(x)

        return f0, f1, f2, f3

    