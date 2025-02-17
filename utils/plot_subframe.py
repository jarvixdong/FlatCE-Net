import matplotlib.pyplot as plt
import os

# 数据准备
L = [8,9,10,11,12] # dBm 横轴数据
SDUNet_64 = [0.098255, 0.036994, 0.01701, 0.010188, 0.007203] 
LS_NMSE = [0.38713, 0.054492, 0.018839, 0.010848, 0.007338]
MMSE_NMSE = [0.051158786, 0.026171018, 0.013618174, 0.009077507, 0.006509192]

FlatCE_Net = [0.194411, 0.183997, 0.127541, 0.144586, 0.093255]
LS_NMSE = [1.690374, 0.779491, 0.311853, 0.322463, 0.166494]
MMSE_NMSE = [0.173299815, 0.178497301, 0.123810392, 0.137731364, 0.090993344]


# 绘图
plt.figure(figsize=(8, 6))
# plt.plot(L, SDUNet_64, 's-', label='SDUNet_64', linewidth=1.5, color='#0072BD')
# plt.plot(L, LS_NMSE, 'd-', label='LS method', linewidth=1.5, color='#77AC30')
# plt.plot(L, MMSE_NMSE, 'p:', label='MMSE method', linewidth=1.5, color='#A2142F')
# plt.plot(L, CDRN_NMSE, 'd-.', label='SIM-Optimized, N=64', linewidth=1.5, color='#77AC30')
plt.plot(L, FlatCE_Net, 's-', label='FlatCE-Net', linewidth=2, color='#1f77b4', markersize=8)  # 蓝色, 方形标记
plt.plot(L, LS_NMSE, 'p:', label='LS method', linewidth=2, color='#ff7f0e', markersize=8)   # 橙色, 圆形标记
plt.plot(L, MMSE_NMSE, '^:', label='MMSE method', linewidth=2, color='#2ca02c', markersize=8)  # 绿色, 三角形标记
# plt.plot(L, CDRN_NMSE, 'd:', label='CDRN method', linewidth=2, color='pink', markersize=8)  # 红色, 叉号标记



# 图例
plt.legend(loc='upper right', fontsize=10)

# plt.yscale('log')  # y 轴取对数
plt.xlabel('S', fontsize=12)
plt.ylabel('Normed MSE (log scale)', fontsize=12)
plt.title('Performance Comparison with Logarithmic Y-Axis', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xticks(range(8, 13, 1))

# 额外的文本标注
# plt.text(5.5, 0.4, 'Solid Lines: N = 32, M = 4', fontsize=10)
# plt.text(5.5, 0.35, 'Dashdotted Lines: N = 64, M = 4', fontsize=10)

# 保存路径
output_folder = '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures'
os.makedirs(output_folder, exist_ok=True)

# 保存为 PNG 和 EPS
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'FlatCENet_Subframe_performance_comparison_nolog.png'), dpi=300)
# plt.savefig(os.path.join(output_folder, 'performance_comparison.eps'), format='eps')
plt.savefig(os.path.join(output_folder, 'FlatCENet_Subframe_performance_comparison_nolog.pdf'), format='pdf')


# 显示图形
# plt.show()
