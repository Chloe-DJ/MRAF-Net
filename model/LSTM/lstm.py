#可以对比MFCC等特征使用CNN和使用LSTM的分类结果，另外考虑CNN-LSTM混合模型，CNN 先提局部模式，LSTM 再学习时间上的依赖关系。
#特征融合、局部全局结合


import torch
import torch.nn as nn
import torch.nn.functional as F

#bidirectional=True = 使用双向 LSTM，能学到前后文信息
class LSTMClassifier(nn.Module):
    def __init__(self, n_mfcc, hidden_size, num_classes, num_layers=2, bidirectional=True, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=n_mfcc,  # 输入维度 = MFCC 特征维度
            hidden_size=hidden_size,  # 隐藏层大小
            num_layers=num_layers,  # LSTM 堆叠层数
            batch_first=True,  # 输入 shape = (batch, time, feature)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 输出全连接层
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, time_steps, n_mfcc)
        if x.dim() == 4:
            x = x.squeeze(1)  # 变成 [B, F, T]
        x = x.permute(0, 2, 1)  # [B, T, F]
        out, (hn, cn) = self.lstm(x)  # out shape: (batch, time, hidden*dir)

        # 取最后时间步的输出 (也可以取 mean/max pooling)
        out_last = out[:, -1, :]  # (batch, hidden*dir)

        logits = self.fc(out_last)  # (batch, num_classes)
        return logits



import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=42, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向GRU → *2

    def forward(self, x):
        # x: (batch, time, n_mfcc)
        out, h_n = self.gru(x)
        # 取最后时间步的 hidden state（拼接双向）
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # (batch, hidden*2)
        logits = self.fc(last_hidden)
        return logits


import torch
import torch.nn as nn


class ConvGRUClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=42, dropout=0.3):
        super(ConvGRUClassifier, self).__init__()

        # 1. 卷积层（处理 n_mfcc 维度 → 提取局部频率模式）
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 时间维度减半

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 再减半
        )

        # 2. GRU 层（时序建模）
        self.gru = nn.GRU(
            input_size=128,  # 卷积输出通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 3. 分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 → *2

    def forward(self, x):
        # x: (batch, time, n_mfcc)
        x = x.permute(0, 2, 1)  # 变为 (batch, n_mfcc, time)，符合 Conv1d 输入格式

        # 卷积特征提取
        conv_out = self.conv(x)  # (batch, channels=128, time')

        # 转换回 RNN 格式 (batch, time', channels)
        conv_out = conv_out.permute(0, 2, 1)

        # GRU
        out, h_n = self.gru(conv_out)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)  # (batch, hidden*2)

        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits
