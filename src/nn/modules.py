import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, 1), padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        self.instnorm_1 = nn.InstanceNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)
        self.instnorm_2 = nn.InstanceNorm2d(out_channels)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
    
    def forward(self, x):
        first_conv = self.conv_1(x)
        first_activation = self.relu_1(self.batchnorm_1(first_conv) + self.instnorm_1(first_conv))
        second_conv = self.conv_2(first_activation)
        second_activation = self.relu_2(self.batchnorm_2(second_conv) + self.instnorm_2(second_conv))

        return self.conv_3(second_activation) + first_conv
    

class NippleModel(nn.Module):
    def __init__(self):
        super(NippleModel, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
 

 