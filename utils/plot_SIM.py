import matplotlib.pyplot as plt
import os

# 数据准备
L = [2,3,4,5,6] # dBm 横轴数据
# SDUNet_64 = [0.398879,0.143324,0.098255, 0.068289, 0.065288] 
# LS_NMSE = [28.059807, 1.239765, 0.38713, 0.199826, 0.196495]
# MMSE_NMSE = [0.12830184, 0.074577348, 0.051158786, 0.041384909, 0.040207503]

FlatCE_Net = [0.284101, 0.24624, 0.194411, 0.216062, 0.206377]
LS_NMSE = [11.096421, 3.560983, 1.690374, 1.9375, 1.768164]
MMSE_NMSE = [0.242319863, 0.220775224, 0.173299815, 0.186430324, 0.190794827]


# SDUNet_64 = [0.143324,0.098255, 0.068289, 0.065288] 
# LS_NMSE = [1.239765, 0.38713, 0.199826, 0.196495]
# MMSE_NMSE = [0.074577348, 0.051158786, 0.041384909, 0.040207503]

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
plt.xlabel('L', fontsize=12)
plt.ylabel('Normed MSE (log scale)', fontsize=12)
plt.title('Performance Comparison with Logarithmic Y-Axis', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xticks(range(2, 7, 1))

# 额外的文本标注
# plt.text(5.5, 0.4, 'Solid Lines: N = 32, M = 4', fontsize=10)
# plt.text(5.5, 0.35, 'Dashdotted Lines: N = 64, M = 4', fontsize=10)

# 保存路径
output_folder = '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures'
os.makedirs(output_folder, exist_ok=True)

# 保存为 PNG 和 EPS
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'FlatCENet_SIM_performance_comparison_nolog.png'), dpi=300)
# plt.savefig(os.path.join(output_folder, 'performance_comparison.eps'), format='eps')
plt.savefig(os.path.join(output_folder, 'FlatCENet_SIM_performance_comparison_nolog.pdf'), format='pdf')


# 显示图形
# plt.show()
