import math
import os

from model.fusion2 import Fusion_model11, Fusion_model12, Fusion_model13, Fusion_model14, ImprovedFusionModel

os.environ['TORCH_HOME'] = '../../pretrained_models'

from torchvision import models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

from model.fusion import Fusion_model1, Fusion_model2, Fusion_model3, Fusion_model

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18_LSTM(nn.Module):
    def __init__(self, num_classes=42, lstm_hidden=256, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()

        # 1️⃣ CNN部分：加载 ResNet18 主干
        base_model = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # 去掉最后的池化和全连接层
        # 输出形状大致为 [B, 512, H/32, W/32]
        # 对输入 [B, 1, 128, 400]，输出大约 [B, 512, 4, 13]

        # 2️⃣ 自适应池化：压缩频率维，保留时间维
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # 输出 [B, 512, 1, T']
        # 这会让我们保留时间维度（T'≈13），方便送入 LSTM

        # 3️⃣ LSTM 模块：建模时间维上的动态变化
        self.lstm = nn.LSTM(
            input_size=512,            # 每个时间步的特征维度
            hidden_size=lstm_hidden,   # LSTM隐层大小
            num_layers=num_layers,     # LSTM层数
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # 4️⃣ 分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 128, 400]
        x = self.features(x)           # [B, 512, H', T']
        x = self.avgpool(x)            # [B, 512, 1, T']
        x = x.squeeze(2)               # [B, 512, T']
        x = x.permute(0, 2, 1)         # [B, T', 512]

        # LSTM层
        out, _ = self.lstm(x)          # [B, T', lstm_out_dim]
        out = out[:, -1, :]            # 取最后一个时间步的输出（也可以做 mean pooling）

        # 分类层
        out = self.classifier(out)
        return out

#写一个简单的四层卷积：
class Audio_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(128 * 10 * 29, 512)  # 全连接层输入是一维向量
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 42)


    def forward(self,x):
        x1 = x.repeat(1, 3, 1, 1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)

        x = x1.view(-1, 128 * 10 * 29)  # output(32*5*5) view函数将向量展平为一维向量，-1是第一个维度，自动推理
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class AlexNet(nn.Module):
    def __init__(self,num_classes=42):
        super().__init__()
        self.features = nn.Sequential(  # nn.Sequential将一系列的层结构进行打包，组合成一个新的结构，
            # 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),  # inplace增加计算量同时降低内存使用
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 13, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x1,x2,x3):
        x1 = x1.repeat(1, 3, 1, 1)  # (B, 1, F, L) -> (B, 3, F, L)
        x2 = x2.repeat(1, 3, 1, 1)
        x3 = x3.repeat(1, 3, 1, 1)
        x1 = self.features(x1)  #以224×448作为输入，得到的输出大小是B，128，6，13
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.classifier(x1)

        x2 = self.features(x2)
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.classifier(x2)

        x3 = self.features(x3)
        x3 = torch.flatten(x3, start_dim=1)
        x3 = self.classifier(x3)
        return (x1 + x2 + x3) / 3

class ResNetClassifier(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        if model_type=='resnet50':
            self.resnet = models.resnet50(pretrained=False)
        elif model_type=='resnet152':
            self.resnet = models.resnet152(pretrained=False)
        elif model_type=='resnet18':
            self.resnet = models.resnet18(pretrained=False)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=42)
        # 去掉最后的全连接层以及pool层，保留特征输出
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fusion_model = ImprovedFusionModel(in_channels=512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, C, 1, 1)
            nn.Flatten(),  # -> (B, C)
            nn.Linear(512, 128),  # C = fusion_model 输出通道数
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 42)  # 输出类别数
        )

    def forward(self, x1, x2, x3):
        # if x.ndim == 2:  # (F, T)
        #     x = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, F, T)
        # elif x.ndim == 3:  # (B, F, T)
        #     x = x.unsqueeze(1)  # -> (B, 1, F, T)
        #     # elif x.ndim == 4:    # (B, C, F, T)，正常情况就不管了
        #
        # print("forward after reshape:", x.shape)
        # x = F.interpolate(x, size=(128, 256), mode="bilinear", align_corners=False)
        # x = x.repeat(1, 3, 1, 1)    # (B, 1, F, L) -> (B, 3, F, L)


        x1 = x1.repeat(1, 3, 1, 1)  # (B, 1, F, L) -> (B, 3, F, L)
        x2 = x2.repeat(1, 3, 1, 1)
        x3 = x3.repeat(1, 3, 1, 1)
        x1 = self.resnet(x1)  # 以224×448作为输入，得到的输出大小是B，128，6，13
        x2 = self.resnet(x2)
        x3 = self.resnet(x3)
        fused_feat = self.fusion_model(x1, x2, x3)
        predictions = self.classifier(fused_feat)



        return predictions




class BasicBlock(nn.Module): #18层和34层对应的残差结构，继承自nn.module
    expansion = 1 #残差结构中主分支所采用的卷积核个数有没有发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):#in_channel, out_channel输入输出特征矩阵深度
        # #downsample对应残差结构中虚线对应的分支；out_channel为主分支上卷积核的个数
        #stride为1对应实线的残差结构，stride为2对应虚线的残差结构；
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module): #50层、101、152层的结构
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block, #定义残差结构，根据不同网络选择bottleneck或basicblock
                 blocks_num,  #对应使用的残差结构的数目，是列表参数形式
                 num_classes=1000, #训练集分类个数
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 #固定，不管是哪个层网络都为64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  #对应conv_2的残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2) #对应conv_3的残差结构
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):#channel残差结构中卷积核使用的卷积个数
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)



#做一个ConvLSTM的baseline，再是试一个vision transformer？ AST (Audio Spectrogram Transformer)、PaSST (Patchout Spectrogram Transformer)

class ConvLSTMCell(nn.Module):
    """单个 ConvLSTM 单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_h, H, W)
        conv_output = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """多层 ConvLSTM"""
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

        # LayerNorm 提升稳定性
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dims[i]) for i in range(num_layers)])

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        h_t, c_t = [], []

        for i in range(self.num_layers):
            h_t.append(torch.zeros(b, self.hidden_dims[i], h, w, device=x.device))
            c_t.append(torch.zeros(b, self.hidden_dims[i], h, w, device=x.device))

        outputs = []
        for step in range(t):
            x_t = x[:, step]
            for i, layer in enumerate(self.layers):
                h_t[i], c_t[i] = layer(x_t, h_t[i], c_t[i])
                # LayerNorm 作用在 (B, C, H, W)
                h_t[i] = self.norms[i](h_t[i].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                x_t = h_t[i]
            outputs.append(h_t[-1])

        outputs = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        return outputs, (h_t, c_t)


class ConvLSTM_FrogClassifier(nn.Module):
    def __init__(self, num_classes=42, in_channels=1):
        super().__init__()
        # CNN 特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (F/2, T/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (F/4, T/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 新增的第四层（比原始多一层）
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (F/16, T/16)
        )

        # 双层 ConvLSTM
        self.convlstm = ConvLSTM(
            input_dim=256,
            hidden_dims=[256, 128],
            kernel_size=3,
            num_layers=2
        )

        # Dropout 增强泛化
        self.dropout = nn.Dropout(0.5)

        # 分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x1,x2,x3):  #分别对应log-mel, stft, lfcc
        # x: (B, 1, F, T)
        B, C, F, T = x1.shape
        features = self.cnn(x1)  # (B, 64, F', T')

        # 转换为 ConvLSTM 输入格式
        features = features.permute(0, 3, 1, 2)  # (B, T', C, F')
        features = features.unsqueeze(3)         # (B, T', C, F', 1)

        out_seq, _ = self.convlstm(features)
        out_last = out_seq[:, -1]  # 取最后时间步
        out_last = self.dropout(out_last)
        out = self.classifier(out_last)
        return out


#VIT模型
# class PatchEmbedAST(nn.Module):
#     """PatchEmbed 适配 log-Mel 输入"""
#     def __init__(self, img_size=(128,371), patch_size=(16,16), in_chans=1, embed_dim=768, stride=(16,9)):
#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = ((img_size[0]-patch_size[0])//stride[0]+1) * ((img_size[1]-patch_size[1])//stride[1]+1)
#
#     def forward(self, x):
#         x = self.proj(x).flatten(2).transpose(1,2)
#         return x
#
# class ViTForLogMel(nn.Module):
#     def __init__(self, label_dim=42, input_fdim=128, input_tdim=371, model_size='base224', imagenet_pretrain=True):
#         super().__init__()
#
#         # 选择 DeiT 模型
#         if model_size == 'base224':
#             self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
#         else:
#             raise ValueError('只支持 base224 作为示例')
#
#         self.original_embedding_dim = self.v.pos_embed.shape[2]
#
#         # 自动计算 tstride
#         patch_kernel = 16
#         desired_patch_time = 40
#         tstride = max(1, (input_tdim - patch_kernel) // (desired_patch_time - 1))
#         fstride = 16  # 频率方向 stride
#
#         # 替换 PatchEmbed
#         self.v.patch_embed = PatchEmbedAST(
#             img_size=(input_fdim, input_tdim),
#             patch_size=(16,16),
#             in_chans=1,
#             embed_dim=self.original_embedding_dim,
#             stride=(fstride,tstride)
#         )
#
#         # 计算 patch 输出
#         test_input = torch.randn(1,1,input_fdim,input_tdim)
#         test_out = self.v.patch_embed.proj(test_input)
#         f_dim, t_dim = test_out.shape[2], test_out.shape[3]
#         num_patches = f_dim * t_dim
#         self.v.patch_embed.num_patches = num_patches
#
#         # 调整位置编码
#         old_pos_embed = self.v.pos_embed[:, 2:, :].detach()
#         old_pos_embed_2d = old_pos_embed.transpose(1,2).reshape(1, self.original_embedding_dim, int(self.v.patch_embed.num_patches**0.5), int(self.v.patch_embed.num_patches**0.5))
#         new_pos_embed_2d = F.interpolate(old_pos_embed_2d, size=(f_dim,t_dim), mode='bilinear')
#         new_pos_embed = new_pos_embed_2d.reshape(1,self.original_embedding_dim,-1).transpose(1,2)
#         self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
#
#         # 分类头
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.original_embedding_dim),
#             nn.Linear(self.original_embedding_dim, label_dim)
#         )
#
#     def forward(self, x):
#         """
#         x: (B, T, F) log-Mel
#         """
#         x = x.unsqueeze(1)       # (B,1,T,F)
#         x = x.transpose(2,3)     # (B,1,F,T)
#         B = x.shape[0]
#
#         x = self.v.patch_embed(x)
#         cls_tokens = self.v.cls_token.expand(B, -1, -1)
#         dist_token = self.v.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#         x = x + self.v.pos_embed
#         x = self.v.pos_drop(x)
#         for blk in self.v.blocks:
#             x = blk(x)
#         x = self.v.norm(x)
#         x = (x[:,0] + x[:,1])/2
#         x = self.mlp_head(x)
#         return x
#
