import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import swanlab
from torchvision import transforms, datasets

def create_dataset_csv():
    # 数据集根目录
    data_dir = './GTZAN/genres_original'
    data = []

    # 遍历所有子目录
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 遍历子目录中的所有wav文件
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    data.append([audio_path, label])

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('audio_dataset.csv', index=False)
    return df


# 自定义数据集类,audio name/path,labels
class AudioDataset(Dataset):
    def __init__(self, df, resize, train_mode=True):
        self.audio_paths = df['path'].values
        # 将标签转换为数值
        self.label_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
        self.labels = [self.label_to_idx[label] for label in df['label'].values]
        self.resize = resize
        self.train_mode = train_mode  # 添加训练模式标志
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])

        # 将音频转换为梅尔频谱图
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=640,
            n_mels=128
        )
        mel_spectrogram = transform(waveform)

        # 确保数值在合理范围内
        mel_spectrogram = torch.clamp(mel_spectrogram, min=0)

        # 转换为3通道图像格式 (为了适配ResNet)
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)

        # 确保尺寸一致
        # resize = torch.nn.AdaptiveAvgPool2d((self.resize, self.resize))
        # mel_spectrogram = resize(mel_spectrogram)
        mel_spectrogram = self.transform(mel_spectrogram)

        return mel_spectrogram, self.labels[idx]

def get_data_loader(batch_size,nw):
    train_data = AudioDataset('E:/pythonProject1/train.txt', transform=transforms.ToTensor())
    test_data = AudioDataset('E:/pythonProject1/test.txt', transform=transforms.ToTensor())
    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    train_num = len(train_data)
    return train_loader,train_num,test_loader
    print('加载成功！')




