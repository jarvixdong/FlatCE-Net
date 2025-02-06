import matplotlib.pyplot as plt
import os

# 数据准备
L = [5,10,15,20,25] # dBm 横轴数据
UNet_NMSE = [0.219331, 0.14436, 0.06597, 0.026025, 0.0092]
SDUNet_64 = [0.166144,0.098255, 0.044228, 0.020033, 0.008632] 
SDUNet_1d = [0.10123, 0.063706, 0.035241, 0.016346, 0.007317]
LS_NMSE = [0.974377, 0.38713, 0.095006, 0.029588, 0.009357]
MMSE_NMSE = [0.084306519, 0.051158786, 0.028259902, 0.011173082,0.00491659]
CDRN_NMSE = [0.656101, 0.38535, 0.094865, 0.029588, 0.009357]


# 绘图
plt.figure(figsize=(8, 6))
# plt.plot(L, SDUNet_64, 's-', label='SDUNet_64', linewidth=1.5, color='#0072BD')
# plt.plot(L, LS_NMSE, 'd-', label='LS method', linewidth=1.5, color='#77AC30')
# plt.plot(L, MMSE_NMSE, 'p:', label='MMSE method', linewidth=1.5, color='#A2142F')
# plt.plot(L, CDRN_NMSE, 'd-.', label='SIM-Optimized, N=64', linewidth=1.5, color='#77AC30')
plt.plot(L, UNet_NMSE, 's:', label='UNet', linewidth=2, color='gray', markersize=8)  # 蓝色, 方形标记
plt.plot(L, SDUNet_64, 'o:', label='SDUNet_64', linewidth=2, color='#1f77b4', markersize=8)  # 蓝色, 方形标记
plt.plot(L, SDUNet_1d, 's-', label='SDUNet_1d', linewidth=2, color='orange', markersize=8)  # 蓝色, 方形标记
plt.plot(L, LS_NMSE, 'p:', label='LS method', linewidth=2, color='#ff7f0e', markersize=8)   # 橙色, 圆形标记
plt.plot(L, MMSE_NMSE, '^:', label='MMSE method', linewidth=2, color='#2ca02c', markersize=8)  # 绿色, 三角形标记
plt.plot(L, CDRN_NMSE, 'd:', label='CDRN method', linewidth=2, color='pink', markersize=8)  # 红色, 叉号标记



# 图例
plt.legend(loc='upper right', fontsize=10)

# 坐标轴与标签
# plt.yscale('log')  # y 轴取对数
plt.xlabel('Power (dBm)', fontsize=12)
plt.ylabel('Normed MSE', fontsize=12)
plt.title('Performance Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.xticks(range(5, 26, 5))

# 额外的文本标注
# plt.text(5.5, 0.4, 'Solid Lines: N = 32, M = 4', fontsize=10)
# plt.text(5.5, 0.35, 'Dashdotted Lines: N = 64, M = 4', fontsize=10)

# 保存路径
output_folder = '/users/elq20xd/workspace/channel_estimation/Channel_estimation/figures'
os.makedirs(output_folder, exist_ok=True)

# 保存为 PNG 和 EPS
plt.tight_layout()
# plt.savefig(os.path.join(output_folder, 'SNR_performance_comparison_log.png'), dpi=300)
# # plt.savefig(os.path.join(output_folder, 'performance_comparison.eps'), format='eps')
# plt.savefig(os.path.join(output_folder, 'SNR_performance_comparison_log.pdf'), format='pdf')
plt.savefig(os.path.join(output_folder, 'SNR_performance_comparison_040225.png'), dpi=300)
# plt.savefig(os.path.join(output_folder, 'performance_comparison.eps'), format='eps')
plt.savefig(os.path.join(output_folder, 'SNR_performance_comparison_040225.pdf'), format='pdf')

# 显示图形
# plt.show()
