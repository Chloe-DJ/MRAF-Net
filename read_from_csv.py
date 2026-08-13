import argparse
import os
import pandas as pd
import torchaudio.transforms as T
import librosa
from torchaudio.transforms import (
    MelSpectrogram, AmplitudeToDB,
    MFCC, LFCC, Resample
)
import torchaudio
from torchvision.transforms import Resize
import torch.nn.functional as F


from data.SCMC import SCMCFeatureExtractor, LPCFeatureExtractor

time_mask = torchaudio.transforms.TimeMasking(
    time_mask_param=60,  # mask up to 60 consecutive time windows
)
freq_mask = torchaudio.transforms.FrequencyMasking(
    freq_mask_param=8,  # mask up to 8 consecutive frequency bins
)

import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class LibrosaCQT:
    def __init__(self, args):
        self.sr = args.target_sample_rate
        self.hop_length = args.hop_length
        self.fmin = args.cqt_fmin
        self.n_bins = args.n_cqt_bins
        self.bins_per_octave = args.cqt_bins_per_octave

    def __call__(self, waveform: torch.Tensor):
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)

        # 转 numpy，确保在 CPU 上
        waveform = waveform.detach().cpu().numpy()

        # 计算 CQT（复数）
        cqt = librosa.cqt(
            waveform,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        # 转幅度谱 -> 分贝谱
        # cqt_mag = np.abs(cqt)
        # cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        # 转成 [1, F, T]，方便 CNN 处理
        return torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)
class LibrosaSpectralContrast:
    def __init__(self, args):
        self.sr = args.target_sample_rate
        self.hop_length = args.sc_hop_length
        self.n_bands = args.n_sc_bands  # 频带数
        self.fmin = args.sc_fmin    # 最低频率

    def __call__(self, waveform: torch.Tensor):
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)

        # 转 numpy，确保在 CPU 上
        waveform = waveform.detach().cpu().numpy()

        # 计算谱对比度
        contrast = librosa.feature.spectral_contrast(
            y=waveform,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bands=self.n_bands,
            fmin=self.fmin
        )  # (n_bands+1, T)

        # 转 tensor 并加 batch 维度
        return torch.tensor(contrast, dtype=torch.float32).unsqueeze(0)  # (1, n_bands+1, T)

class AnuraSet(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 args,
                 train=True,
                 ):

        if isinstance(annotations_file, str):
            self.df = pd.read_csv(annotations_file)
        else:
            self.df = annotations_file.copy()

        if train:
            self.df = self.df[self.df["subset"] == "train"]
        else:
            self.df = self.df[self.df["subset"] == "test"]

        # 预处理文件名
        self._preprocess_filenames()
        self.annotations = self.df
        self.audio_dir = audio_dir
        self.args = args
        self.train = train  # 标记是否为训练模式（用于控制增强）

        self._init_transforms() #"""根据args初始化所有特征提取器"""
        self._init_augmentations()# 时间掩码和频率掩码

    def _preprocess_filenames(self):
        """预处理生成新的文件名"""
        self.df["split_name1"] = self.df["fname"].str.split("_").str[0]
        self.df["split_name2"] = self.df["fname"].str.split("_").str[1]
        self.df["split_name3"] = self.df["fname"].str.split("_").str[2]
        self.df["data_start"] = self.df["min_t"].astype(str)
        self.df["data_end"] = self.df["max_t"].astype(str)

        self.df["new_filename"] = (
                self.df["split_name1"] + "_" +
                self.df["split_name2"] + "_" +
                self.df["split_name3"] + "_" +
                self.df["data_start"] + "_" +
                self.df["data_end"] + ".wav"
        )

    def _init_transforms(self):
        """根据args初始化所有特征提取器"""
        # 采样率转换
        self.resampler = Resample(
            orig_freq=22050,
            new_freq=self.args.target_sample_rate# 必须设为 16000（Wav2vec 2.0 预训练采样率）
        )

        # 1. 梅尔频谱图
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.args.target_sample_rate,
            n_fft=self.args.n_fft,
            # win_length=self.args.win_length,
            hop_length=self.args.hop_length,
            n_mels=self.args.n_mels
        )
        self.amplitude_to_db = AmplitudeToDB()

        # 2. MFCC特征
        self.mfcc = MFCC(
            sample_rate=self.args.target_sample_rate,
            n_mfcc=self.args.n_mfcc,
            melkwargs={
                "n_fft": self.args.n_fft,
                "hop_length": self.args.hop_length,
                "n_mels": self.args.n_mels
            }
        )

        # 3. HFCC特征
        self.hfcc = LFCC(
            sample_rate=self.args.target_sample_rate,
            n_lfcc=self.args.n_hfcc,
            speckwargs={
                "n_fft": self.args.n_fft,
                "hop_length": self.args.hop_length
            }
        )
        #4， CQT特征
        self.cqt = LibrosaCQT(self.args)
        # CQT的分贝转换
        self.cqt_amplitude_to_db = AmplitudeToDB()

        # 5. Spectral Contrast特征提取器
        self.spectral_contrast =LibrosaSpectralContrast(self.args)
        self.spectral_contrast_amplitude_to_db = AmplitudeToDB()
        # self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
        #     self.args.wav2vec2_pretrained_model  # 从 args 传入预训练模型名
        # )

        #6 PSD功率谱密度特征 PSD 特征原始大小 = [freq=257, time=187]
        self.psd=torchaudio.transforms.Spectrogram(
            pad=self.args.scd_pad,
            win_length=self.args.scd_window,
            n_fft=self.args.scd_n_fft,
            hop_length=self.args.scd_hop_length,
            power=self.args.scd_power,
            normalized=True
        )

        #7.LFCC 线性频率倒谱系数
        self.lfcc=LFCC(
            sample_rate=self.args.target_sample_rate,
            n_lfcc=self.args.n_lfcc,
            speckwargs={"n_fft": self.args.n_fft, "hop_length": self.args.hop_length, "win_length": self.args.lfcc_win_length}
        )
        # 8 PLP感知线性预测系数  (297, 20)
        # self.plp = speechpy.feature.plp(
        #     samplerate=self.args.target_sample_rate,
        #     frame_length=self.args.plp_winlen,
        #     frame_stride=self.args.plp_winstep,
        #     num_cepstral=self.args.n_plp_cep,
        #     pre_emphasis=self.args.n_plp_cep,
        #     fft_length=self.args.n_fft,
        #     filterbank_channel_count=self.args.plp_nfilts
        # )

        #9.SCMC频谱质心幅度倒谱系数 (B, 21, 258)->(32, 256)
        # self.scmc=SCMCFeatureExtractor(sample_rate=self.args.target_sample_rate,
        #                                n_fft=self.args.n_fft,
        #                                hop_length=self.args.hop_length,
        #                                normalize=True
        #                                )
        # 10.LPC（Linear Predictive Coding，线性预测编码） LPC 可以和 MFCC/PLP/SCMC 拼接做特征增强。
        self.lpc = LPCFeatureExtractor()


        #11 stft生成谱图 [257, 371]
        self.stft=T.Spectrogram(
            n_fft=self.args.n_fft,
            hop_length=self.args.hop_length,
            win_length=self.args.n_fft,
            power=2.0
        )

    def _init_augmentations(self):
        """初始化数据增强变换（仅训练时使用）"""
        # 时间掩码和频率掩码
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=self.args.time_mask_param  # 从args读取参数，灵活调整
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.args.freq_mask_param  # 从args读取参数
        )
        # Wav2vec 2.0 波形增强（训练时添加轻微噪声，提升抗噪性）
        # self.wav2vec2_noise_aug = torchaudio.transforms.AddGaussianNoise(
        #     mean=0.0, std=self.args.wav2vec2_noise_std  # 噪声强度从 args 控制，建议 0.005~0.01
        # )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 获取音频路径并加载音频
        audio_path = self._get_audio_path(index)
        waveform, sample_rate = torchaudio.load(audio_path)

        # 统一采样率
        if sample_rate != self.args.target_sample_rate:
            self.resampler.orig_freq = sample_rate
            waveform = self.resampler(waveform)
        # 3. 统一为单通道（Wav2vec 2.0 输入为单通道）
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]  # 取第一通道，shape: [1, T]

        # 提取多种特征
        mel_spec_3ch, lfcc_feat, stft_feat = self._extract_features(waveform)

        # 获取标签
        label = self._get_label(index)

        return (mel_spec_3ch, lfcc_feat, stft_feat), label, index

    def _apply_augmentations(self, feature):
        """对特征应用掩码增强（仅训练时调用）"""
        # 先时间掩码，再频率掩码
        augmented = self.time_mask(feature)
        augmented = self.freq_mask(augmented)
        return augmented

    def _process_feature(self, feature, need_clamp=True):
        """
        通用特征处理函数：统一处理为3通道格式
        Args:
            feature: 原始特征张量 (shape: [C, D, T])
            need_clamp: 是否需要将数值限制为非负
        Returns:
            处理后的3通道特征张量 (shape: [3, D, T])
        """
        # 确保单通道输入
        if feature.shape[0] > 1:
            feature = feature[0:1, :, :]

        # 数值限制（根据特征类型选择）
        # if need_clamp:
        #     feature = torch.clamp(feature, min=0)

        # 扩展为3通道
        # resize = Resize((224, 448))
        # resize = Resize((256, 32))
        #convlstm做log-mei谱图建议缩放到(128, 256)
        resize = Resize((128, 256))
        feature = resize(feature)
        return feature

    def _extract_features(self, waveform):
        """提取多种音频特征并返回字典"""
        # 确保波形是单通道
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        # 1. 梅尔频谱图 (转为分贝值)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        # 训练时应用增强（仅对梅尔频谱，可根据需求扩展到其他特征）
        if self.train:
            mel_spec_db = self._apply_augmentations(mel_spec_db)
        mel_spec_3ch = self._process_feature(mel_spec_db, need_clamp=True)

        # 2. MFCC特征的形状为 (时间步数, 特征维度)，特征维度 = n_mfcc
        #时间步数 = floor((音频总采样点数 - n_fft) / hop_length) + 1
        #特征维度 = n_mfcc（通常 13-40）  （372，40）
        # mfcc_feat = self.mfcc(waveform)
        # mfcc_3ch = self._process_feature(mfcc_feat, need_clamp=False)

        # 3. HFCC特征，加一个CQT特征？CQT抗噪声，试下CQT+MFCC，CQT 的空间结构更适合 CNN 的卷积操作，且计算效率高于 STFT
        # hfcc_feat = self.hfcc(waveform)
        # hfcc_3ch = self._process_feature(hfcc_feat, need_clamp=False)

        #4. CQT特征
        # cqt_feat=self.cqt(waveform)
        # cqt_mag = torch.abs(cqt_feat)  # 取幅度
        # cqt_db = self.cqt_amplitude_to_db(cqt_mag)  # 转为分贝

        # 归一化
        # mean = cqt_db.mean()
        # std = cqt_db.std() + 1e-6
        # cqt_db = (cqt_db - mean) / std
        #
        # if self.train:
        #     cqt_db = self._apply_augmentations(cqt_db)  # 应用增强


        # 5. Spectral Contrast特征
        # SpectralContrast返回形状为 (n_bands + 1, time_steps)
        # 其中最后一维是全局谷值，我们取前n_bands维
        # sc_feat = self.spectral_contrast(waveform)
        # sc_db = self.spectral_contrast_amplitude_to_db(sc_feat)  # 转为分贝
        # if self.train:
        #     sc_feat = self._apply_augmentations(sc_db)  # 应用增强

        #6.PSD特征 功率谱密度 用CNN做
        # psd_feat = self.psd(waveform)
        # psd_feat_db = self.amplitude_to_db(psd_feat)
        # if self.train:
        #     psd_feat_db = self._apply_augmentations(psd_feat_db)
        # psd_3ch = self._process_feature(psd_feat_db, need_clamp=True)

        #7. LFCC特征 可以分别尝试resnst18和LSTM
        lfcc_feat=self.lfcc(waveform)
        if self.train:
            lfcc_feat = self._apply_augmentations(lfcc_feat)
        lfcc_feat = self._process_feature(lfcc_feat, need_clamp=True)

        #8.PLP 感知线性预测系数  resize到64，256
        # plp_feat = extract_plp_features(waveform)
        # if self.train:
        #     plp_feat = self._apply_augmentations(plp_feat)

        #9.SCMC频谱质心幅度倒谱系数 单独用 SCMC 的研究较少，但作为 补充特征 会提高区分度。
        # scmc_feat=self.scmc(waveform)
        # if self.train:
        #     scmc_feat = self._apply_augmentations(scmc_feat)

        #10. LPC线性预测编码 (298, 16)
        # lpc_feat =self.lpc(waveform)
        # if self.train:
        #     lpc_feat = self._apply_augmentations(lpc_feat)
        # lpc_feat = self._process_feature(lpc_feat, need_clamp=False)

        #11 STFT生成谱图
        stft_feat = self.stft(waveform)
        stft_feat_db = self.amplitude_to_db(stft_feat)
        if self.train:
            stft_feat_db = self._apply_augmentations(stft_feat_db)
        stft_feat= self._process_feature(stft_feat_db, need_clamp=True)




        # 2.1 训练时添加波形噪声增强（仅对 Wav2vec 2.0 输入生效）
        # wav2vec2_waveform = waveform.clone()  # 避免修改原始波形
        # if self.train:
        #     wav2vec2_waveform = self.wav2vec2_noise_aug(wav2vec2_waveform)

        # 2.2 用 Wav2vec 2.0 处理器预处理波形：
        # - 归一化（将波形幅值缩放到 [-1, 1]）
        # - 转换为模型输入格式（返回 dict，含 input_values）
        # wav2vec2_inputs = self.wav2vec2_processor(
        #     wav2vec2_waveform.squeeze(0).numpy(),  # 转为 1D numpy 数组（处理器要求）
        #     sampling_rate=self.args.target_sample_rate,  # 必须为 16000
        #     return_tensors="pt",  # 返回 PyTorch 张量
        #     #保持同意固定长度输入不需要做padding和truncation
        #     padding=False,  # 暂不 padding（batch 级 padding 在 DataLoader 中处理）
        #     truncation=False,  # 超过 max_duration 的波形截断（避免显存溢出）
        #     # max_duration=self.args.wav2vec2_max_duration  # 最大音频时长（如 5.0 秒）
        # )

        # 提取 input_values（shape: [1, T]），移除 batch 维（后续 DataLoader 会加）
        # wav2vec2_input= wav2vec2_inputs["input_values"].squeeze(0)
        # attention_mask=wav2vec2_inputs["attention_mask"].squeeze(0),

        # ============ early concat: 在 channel 维拼接 ============
        # feat = torch.cat([mel_spec_3ch, stft_feat, lfcc_feat], dim=0)

        return mel_spec_3ch, lfcc_feat, stft_feat

    # 可以在 _get_audio_path 方法中添加检查
    def _get_audio_path(self, index):
        split_name1 = self.df.iloc[index]["split_name1"]  # 新增：获取split_name1
        filename = self.df.iloc[index]["new_filename"]
        audio_path = os.path.join(self.audio_dir, split_name1, filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        return audio_path

    def _get_label(self, index):
        values = self.annotations.iloc[index, 8:50].values
        try:
            values = values.astype(float)
        except Exception as e:
            print(f"数据不是数值型: {values}, 错误: {e}")
        return torch.tensor(values, dtype=torch.float32)


def get_data_loader(args):
    """创建并返回训练和测试数据加载器"""
    # 初始化训练集和测试集
    train_dataset = AnuraSet(
        annotations_file=args.annotations_path,
        audio_dir=args.audio_dir,
        args=args,
        train=True
    )

    test_dataset = AnuraSet(
        annotations_file=args.annotations_path,  # 测试集标注文件路径
        audio_dir=args.audio_dir,  # 假设测试音频与训练音频在同一目录
        args=args,
        train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 加速GPU传输
        drop_last = True  # 丢弃最后一个不足batch_size的批次
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 测试集通常不打乱
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last = False  # 丢弃最后一个不足batch_size的批次
    )

    print(f"数据加载成功！训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    return train_loader, len(train_dataset), len(test_dataset),test_loader


