import re
import matplotlib.pyplot as plt
import numpy as np
import os

# def extract_nmse_and_plot(log_path, dst_path="nmse_curve.png"):
#     with open(log_path, 'r') as f:
#         lines = f.readlines()

#     nmse_list = []
#     for line in lines:
#         match = re.search(r'NMMSE:\s*([0-9.]+)', line)
#         if match:
#             nmse_list.append(float(match.group(1)))

#     print("nmse_values:", nmse_list)

#     # 画图
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, len(nmse_list)+1), nmse_list, marker='o', label="NMSE (linear)")
#     plt.xlabel("Epoch")
#     plt.ylabel("NMSE")
#     plt.title("NMSE Convergence Curve")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(dst_path)
#     print(f"Plot saved to {dst_path}")

#     return nmse_list

# def extract_metrics_and_plot(log_path, save_dir="./plots"):
#     os.makedirs(save_dir, exist_ok=True)

#     loss_list = []
#     train_mmse_list = []
#     nmse_list = []

#     with open(log_path, 'r') as f:
#         lines = f.readlines()

#     for line in lines:
#         loss_match = re.search(r'Loss:\s*([0-9.]+)', line)
#         train_match = re.search(r'Train_MMSE:\s*([0-9.]+)', line)
#         nmse_match = re.search(r'NMMSE:\s*([0-9.]+)', line)

#         if loss_match:
#             loss_list.append(float(loss_match.group(1)))
#         if train_match:
#             train_mmse_list.append(float(train_match.group(1)))
#         if nmse_match:
#             nmse_list.append(float(nmse_match.group(1)))

#     loss_list = loss_list[5:50]
#     train_mmse_list = train_mmse_list[5:50]
#     nmse_list = nmse_list[5:50]
#     epochs = range(1, len(loss_list) + 1)

#     def save_plot(data, title, ylabel, filename, marker='o-', label=None):
#         plt.figure(figsize=(10, 4))
#         plt.plot(epochs, data, marker, label=label or ylabel)
#         plt.xlabel("Epoch")
#         plt.ylabel(ylabel)
#         plt.title(title)
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, filename))
#         plt.close()

#     save_plot(loss_list, "Loss Convergence Curve", "Loss", "loss_curve.png")
#     save_plot(train_mmse_list, "Train_MMSE Convergence Curve", "Train_MMSE", "train_mmse_curve.png")
#     save_plot(nmse_list, "NMMSE Convergence Curve", "NMMSE", "nmse_curve.png")

#     print(f"Saved plots to: {save_dir}")
#     return loss_list, train_mmse_list, nmse_list


def extract_metrics_and_plot(log_path, save_dir="./plots", start_epoch=0, end_epoch=None):
    import os
    import re
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    loss_list = []
    train_mmse_list = []
    nmse_list = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        loss_match = re.search(r'Loss:\s*([0-9.]+)', line)
        train_match = re.search(r'Train_MMSE:\s*([0-9.]+)', line)
        nmse_match = re.search(r'NMMSE:\s*([0-9.]+)', line)

        if loss_match:
            loss_list.append(float(loss_match.group(1)))
        if train_match:
            train_mmse_list.append(float(train_match.group(1)))
        if nmse_match:
            nmse_list.append(float(nmse_match.group(1)))

    # 原始 epoch 编号（从 1 开始）
    epochs = list(range(1, len(loss_list) + 1))

    # 如果没指定 end_epoch，则默认为结尾
    if end_epoch is None:
        end_epoch = len(loss_list)

    # 截取中间区段
    loss_list = loss_list[start_epoch:end_epoch]
    train_mmse_list = train_mmse_list[start_epoch:end_epoch]
    nmse_list = nmse_list[start_epoch:end_epoch]
    epochs = epochs[start_epoch:end_epoch]

    def save_plot(data, title, ylabel, filename, marker='o-', label=None):
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, data, marker, label=label or ylabel)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        # plt.yticks(np.arange(0.0130, 0.0153, 0.0003))
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    save_plot(loss_list, "Loss Convergence Curve", "Loss", "loss_curve.png")
    save_plot(train_mmse_list, "Train_MMSE Convergence Curve", "Train_MMSE", "train_mmse_curve.png")
    save_plot(nmse_list, "NMMSE Convergence Curve", "NMMSE", "nmse_curve.png")

    # 合并图
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_mmse_list, 'o-', label="Train_MMSE")
    plt.plot(epochs, nmse_list, 's--', label="NMMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Train_MMSE & NMMSE (Clipped)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_and_nmse_curve.png"))
    plt.close()

    print(f"Saved plots to: {save_dir}")
    return loss_list, train_mmse_list, nmse_list


# # 使用方法（替换为你的日志路径）
# src = '/mnt/parscratch/users/elq20xd/channel_estimation/save_models/fixed_UEs_15GHz_ReduceLR/valid_Dataset_dB-15_N36_K4_L4_S13_Setup20_Reliz1000Ran1_flatCE_l2c32dia6/train.log'
# dst = '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/training_processing/valid_Dataset_dB-15_N36_K4_L4_S13_Setup20_Reliz1000Ran1_flatCE_l2c32dia6'

method = 'valid_Dataset_dB-15_N36_K4_L4_S13_Setup20_Reliz1000Ran1_flatCE_l2c32dia6_sp'
# method = 'valid_Dataset_dB-15_N36_K4_L4_S13_Setup20_Reliz1000Ran1_flatCE_l2c32dia6'
# method = 'valid_Dataset_dB-15_N36_K4_L4_S13_Setup20_Reliz1000Ran1_DnCNN_MultiBlock_ds'
src = '/mnt/parscratch/users/elq20xd/channel_estimation/save_models/fixed_UEs_15GHz_ReduceLR/%s/train.log'%method
dst = '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/training_processing/%s'%method
# dst = '/mnt/parscratch/users/elq20xd/channel_estimation/training_processing/%s'%method
nmse_values = extract_metrics_and_plot(src,dst,0,50)