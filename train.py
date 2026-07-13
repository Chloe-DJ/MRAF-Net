from __future__ import print_function
import argparse
import glob
import sys
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms, datasets, models
import warnings

from data.read_from_csv import get_data_loader
from model.Audio_model import resnet50, ResNetClassifier, Audio_model
from utils.draw_loss_and_acuracy import draw_loss_and_accuracy
from utils.evaluation_metrics import MetricTracker, compute_sample_wise_accuracy, compute_macro_auroc, \
    compute_subset_accuracy, compute_mean_average_precision, Compute_hamming_loss, Compute_micro_f1, \
    Compute_labelwise_f1, Compute_ranking_loss, Compute_macro_f1

from utils.parameter import parser_args
import random
warnings.filterwarnings("ignore", category=UserWarning)
import os

def get_args():
    args = parser_args()
    return args

def main():
    args = get_args()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print("==========\nArgs:{}\n==========".format(args))
    print('==> Loading data..')
    print('==> Building model..')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)

    # net=resnet50(num_classes=42, include_top=True)
    net=ResNetClassifier("resnet18")
    state = torch.load('resnet18-5c106cde.pth', map_location='cpu')
    # 排除 fc 层
    state = {k: v for k, v in state.items() if not k.startswith('fc.')}
    missing, unexpected = net.resnet.load_state_dict(state, strict=False)
    print('missing:', missing, 'unexpected:', unexpected)
    net.to(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


    criterion_id =nn.BCEWithLogitsLoss().to(device)
    #试一下ASL loss

    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16  # 从参数获取或默认
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 工作进程数
    # print(f'Using {nw} dataloader workers every process')

    train_loader,train_num,test_num,test_loader=get_data_loader(args)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr if hasattr(args, 'lr') else 0.0004)  # 从参数获取学习率
    # optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs =args.epochs
    best_acc = 0.0
    total = 0
    correct = 0
    save_path = '/public/home/sklec04/Frog_audio_classification/resnet18_classification.pth'
    train_steps = len(train_loader)
    print("start training")

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        train_losses = MetricTracker()
        train_sample_wise_accuracy = MetricTracker()
        train_macro_auroc = MetricTracker()
        train_subset_accuracy = MetricTracker()
        train_mean_average_precision = MetricTracker()
        train_hamming_loss = MetricTracker()
        train_ranking_loss = MetricTracker()
        train_macro_f1 = MetricTracker()
        train_labelwise_f1 = MetricTracker()
        train_label_f1_list = []  # 存储每个类别的F1

        iters = 0
        # train
        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (inputs,label,index) in enumerate(train_bar):
            images1= inputs.to(device)
            labels = label.to(device)
            out=net(images1.to(device))
            loss=criterion_id(out.to(device),labels.to(device))
            # 将 out 转为 sigmoid 概率
            y_pred = torch.sigmoid(out)
            # loss_m = loss_function2(out0.to(device), labels.to(device))
            # train_acc =train_acc+ compute_sample_wise_accuracy(labels,y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新损失指标
            train_losses.update(loss.item())

            #计算并更新指标
            train_sample_wise_accuracy.update(compute_sample_wise_accuracy(labels, y_pred))
            train_macro_auroc.update(compute_macro_auroc(labels, y_pred))
            train_subset_accuracy.update(compute_subset_accuracy(labels, y_pred))
            train_mean_average_precision.update(compute_mean_average_precision(labels, y_pred))
            train_hamming_loss.update(Compute_hamming_loss(labels, y_pred))
            train_ranking_loss.update(Compute_ranking_loss(labels, y_pred))
            train_macro_f1.update(Compute_macro_f1(labels, y_pred))
            # label-wise f1 可以保存为 array 或绘图
            labelwise_f1_score = Compute_labelwise_f1(labels, y_pred)
            train_label_f1_list.append(labelwise_f1_score)
            train_labelwise_f1.update(np.mean(labelwise_f1_score))

            if step % 100== 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Step [{step}/{len(train_loader)}]  \
                       Loss: {loss.item():.4f}  \
                       SampleAcc: {train_sample_wise_accuracy.avg:.4f}  \
                       SubsetAcc: {train_subset_accuracy.avg:.4f}  \
                       mAP: {train_mean_average_precision.avg:.4f}  \
                       HammingLoss: {train_hamming_loss.avg:.4f}  \
                       RankLoss: {train_ranking_loss.avg:.4f}  \
                       MacroF1: {train_macro_f1.avg:.4f}  \
                       MacroAUROC: {train_macro_auroc.avg:.4f}  \
                       LabelwiseF1(mean): {train_labelwise_f1.avg:.4f}")
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 计算每个类别的平均F1
        avg_train_label_f1 = np.mean(train_label_f1_list, axis=0)
        # scheduler.step()
            # validate
        print("start validate")

        # 初始化验证指标（建议与训练使用相同的结构）
        val_losses = MetricTracker()
        val_sample_wise_accuracy = MetricTracker()
        val_macro_auroc = MetricTracker()
        val_subset_accuracy = MetricTracker()
        val_mean_average_precision = MetricTracker()
        val_hamming_loss = MetricTracker()
        val_ranking_loss = MetricTracker()
        val_macro_f1 = MetricTracker()
        val_labelwise_f1 = MetricTracker()
        val_label_f1_list = []

        print("Start validation...")
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for step, (val_input,val_label,index) in enumerate(val_bar):
                val_images1= val_input.to(device)
                val_labels = val_label.to(device)
                val_out=net(val_images1.to(device))
                loss_val = criterion_id(val_out.to(device), val_labels.to(device))
                predict_y = torch.sigmoid(val_out)

                # 更新损失
                val_losses.update(loss_val.item())
                # 更新评价指标
                val_sample_wise_accuracy.update(compute_sample_wise_accuracy(val_labels, predict_y))
                val_macro_auroc.update(compute_macro_auroc(val_labels, predict_y))
                val_subset_accuracy.update(compute_subset_accuracy(val_labels, predict_y))
                val_mean_average_precision.update(compute_mean_average_precision(val_labels, predict_y))
                val_hamming_loss.update(Compute_hamming_loss(val_labels, predict_y))
                val_ranking_loss.update(Compute_ranking_loss(val_labels, predict_y))
                val_macro_f1.update(Compute_macro_f1(val_labels, predict_y))
                # 记录每个类别的F1分数
                val_labelwise_f1_score = Compute_labelwise_f1(val_labels.cpu(), predict_y.cpu())
                val_label_f1_list.append(val_labelwise_f1_score)
                val_labelwise_f1.update(np.mean(val_labelwise_f1_score))

                if step % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}] Step [{step}/{len(test_loader)}]  \
                                           Loss: {loss_val.item():.4f}  \
                                           SampleAcc: {val_sample_wise_accuracy.val:.4f}  \
                                           SubsetAcc: {val_subset_accuracy.val:.4f}  \
                                           mAP: {val_mean_average_precision.val:.4f}  \
                                           HammingLoss: {val_hamming_loss.val:.4f}  \
                                           RankLoss: {val_ranking_loss.val:.4f}  \
                                           MacroF1: {val_macro_f1.val:.4f}  \
                                           MacroAUROC: {val_macro_auroc.val:.4f}  \
                                           LabelwiseF1(mean): {val_labelwise_f1.val:.4f}")
        # 计算验证集每个类别的平均F1
        avg_val_label_f1 = np.mean(val_label_f1_list, axis=0)

        # 打印 epoch 总结
        print(
            f'========================= Epoch : [{epoch + 1}/{epochs}]  Evaluation ======================================================================')

        print("Training Results : ")
        print(
            f"Training Loss        : {train_losses.avg:.4f}, Training Subset Accuracy : {train_subset_accuracy.avg:.4f},"
            f" Training mean_average_precision : {train_mean_average_precision.avg:.4f}")
        print("Training Label-wise F1 (per class):")
        for i, avg_f1 in enumerate(avg_train_label_f1):
            print(f"    Class {i}: {avg_f1:.4f}")

        print("\nValidation Results : ")
        print(
            f"Validation Loss      : {val_losses.avg:.4f}, Validation Subset Accuracy : {val_subset_accuracy.avg:.4f}, Validation F1-macro : {val_macro_f1.avg:.4f},"
            f"Validation mean_average_precision : {val_mean_average_precision.avg:.4f},Validation hamming_loss : {val_hamming_loss.avg:.4f},Validation ranking_loss : {val_ranking_loss.avg:.4f}")
        print("Validation Label-wise F1 (per class):")
        for i, avg_f1 in enumerate(avg_val_label_f1):
            print(f"    Class {i}: {avg_f1:.4f}")

        # 记录指标
        train_loss_list.append(train_losses.avg)
        val_loss_list.append(val_losses.avg)
        train_acc_list.append(train_macro_f1.avg)
        val_acc_list.append(val_macro_f1.avg)

        print(
                f'[epoch {epoch + 1}] train_loss: {train_losses.avg:.3f}  train_accuracy: {train_sample_wise_accuracy.avg:.3f}  val_accuracy: {val_sample_wise_accuracy.avg:.3f}')

        # 保存最佳模型
        if val_macro_f1.avg > best_acc:
            best_acc = val_macro_f1.avg
            torch.save(net.state_dict(), save_path)
            print(f"Saved best model with accuracy: {best_acc:.4f}")

        # 保存指标到文件
        save_dir = '/public/home/sklec04/Frog_audio_classification'
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

        with open(os.path.join(save_dir, 'train_resnet18.txt'), 'w') as f:
            for item in train_acc_list:
                f.write(f"{item}\n")

        with open(os.path.join(save_dir, 'val_resnet18.txt'), 'w') as f:
            for item in val_acc_list:
                f.write(f"{item}\n")

        with open(os.path.join(save_dir, 'train_resnet18_loss.txt'), 'w') as f:
            for item in train_loss_list:
                f.write(f"{item}\n")

        with open(os.path.join(save_dir, 'val_resnet18_loss.txt'), 'w') as f:
            for item in val_loss_list:
                f.write(f"{item}\n")
    torch.cuda.empty_cache()

    # draw_loss_and_accuracy(train_loss_list, val_loss_list, val_acc_list, train_acc_list, epochs)

    print('Finished Training')


if __name__ == '__main__':

    main()









