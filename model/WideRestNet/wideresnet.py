import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 基本残差块
# ===============================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, dropRate=0.0):
        super(ResBlock, self).__init__()
        self.equalInOut = (in_channels == out_channels) and (stride == 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropRate = dropRate

        self.shortcut = None
        if not self.equalInOut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = x if self.equalInOut else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


# ===============================
# 残差堆栈：1个Downsample + 2个ResBlock
# ===============================
class ResStack(nn.Module):
    def __init__(self, in_channels, out_channels, dropRate=0.0):
        super(ResStack, self).__init__()
        self.down = ResBlock(in_channels, out_channels, stride=2, dropRate=dropRate)
        self.block1 = ResBlock(out_channels, out_channels, dropRate=dropRate)
        self.block2 = ResBlock(out_channels, out_channels, dropRate=dropRate)

    def forward(self, x):
        x = self.down(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


# ===============================
# 主干网络
# ===============================
class WideResNet(nn.Module):
    def __init__(self, num_classes=42, dropRate=0.3):
        super(WideResNet, self).__init__()

        # Pre-processing
        self.conv_pre = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)  # 输入 (1,64,384) → (32,64,384)
        self.bn_pre = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool_pre = nn.MaxPool2d(kernel_size=2, stride=2)  # (32,64,384) → (32,64,192)

        # ResStacks
        self.stack1 = ResStack(32, 64, dropRate=dropRate)     # → (64,32,96)
        self.stack2 = ResStack(64, 128, dropRate=dropRate)    # → (128,16,48)
        self.stack3 = ResStack(128, 256, dropRate=dropRate)   # → (256,8,24)
        self.stack4 = ResStack(256, 512, dropRate=dropRate)   # → (512,4,12)

        # 分类头
        self.head_conv1 = nn.Conv2d(512, 512, kernel_size=(4,10))   # (512,4,12) → (512,1,3)
        self.head_bn1 = nn.BatchNorm2d(512)
        self.head_conv2 = nn.Conv2d(512, 1024, kernel_size=1)       # (512,1,3) → (1024,1,3)
        self.head_bn2 = nn.BatchNorm2d(1024)
        self.head_conv3 = nn.Conv2d(1024, num_classes, kernel_size=1)       # (1024,1,3) → (987,1,3)
        self.head_bn3 = nn.BatchNorm2d(num_classes)

        # Global LME pooling → (987,1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        # Pre-processing
        x = self.conv_pre(x)
        x = self.bn_pre(x)
        x = self.relu(x)
        x = self.pool_pre(x)

        # ResStacks
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        # 分类头
        x = self.head_conv1(x)
        x = self.relu(self.head_bn1(x))
        x = self.head_conv2(x)
        x = self.relu(self.head_bn2(x))
        x = self.head_conv3(x)
        x = self.head_bn3(x)

        # Global pooling
        x = self.global_pool(x)  # (B,987,1,1)
        x = x.view(x.size(0), -1)

        return x

