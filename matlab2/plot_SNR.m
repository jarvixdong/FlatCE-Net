% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [5 10 15 20 25 30];

% ------------------------------paper data 13/03---------------------------
% LS_NMSE = [1.690374, 1.004105, 0.1768, 0.062682, 0.01751];
% MMSE_NMSE = [0.173299815, 0.137608864, 0.05564, 0.030899951, 0.012977496];
% FlatCE_Net = [0.194411, 0.15893, 0.060751, 0.036471, 0.013426];
% CDRN_NMSE = [0.216069, 0.187047, 0.066331, 0.04531, 0.016515];

% --------------------15dBm N4L4S9 is Sutup20Rzl1000---------------------------
% LS_NMSE = [1.64718, 0.4829, 0.17019, 0.04062, 0.01275];
% MMSE_NMSE = [0.18132, 0.10282, 0.0524, 0.02446, 0.01009];
% FlatCE_Net = [0.19899, 0.118893, 0.059984, 0.025657, 0.010624];
% CDRN_NMSE = [0.21476, 0.149353, 0.076029, 0.030888, 0.011644];

% --------------------15dBm N4L4S9 is Sutup10Rzl1000---------------------------
% LS_NMSE = [1.64718, 0.4829, 0.149892, 0.04062, 0.01275];
% MMSE_NMSE = [0.18132, 0.10282, 0.0516345, 0.02446, 0.01009];
% FlatCE_Net = [0.19899, 0.118893, 0.054581, 0.025657, 0.010624];
% CDRN_NMSE = [0.21476, 0.149353, 0.06501, 0.030888, 0.011644];

% --------------------latest fixed position---------------------------
% LS_NMSE = [1.67262	0.50766	0.17983	0.04971	0.01558];
% MMSE_NMSE = [0.17493	0.10181	0.05711	0.0272	0.01171];
% FlatCE_Net = [0.192608	0.117309	0.059814	0.028213	0.011943];
% CDRN_NMSE = [0.209734	0.129529	0.064356	0.030287	0.012892];

% --------------------15GHz fixed position---------------------------

LS_NMSE =[1.85976	0.58811	0.17608	0.06287	0.01632 0.00549];
MMSE_NMSE =[0.18635	0.10907	0.0582	0.02804	0.01187 0.00467];
FlatCE_Net = [0.199116455	0.116891099	0.059325765	0.032278498	0.012053745 0.00471121624520258];
CDRN_NMSE = [0.20400721	0.118668223	0.06073253	0.033121129	0.013052685 0.00506180812151129];


fig = figure('Visible', 'off');
hold on; box on;
% 2) LS_NMSE：虚线 + 蓝色 + 三角标记
plot(p, LS_NMSE, 'b--^','LineWidth',1.5, 'MarkerSize',8);

% 4) CDRN_NMSE：虚线 + 洋红色 + 菱形标记
plot(p, CDRN_NMSE, 'm--d','LineWidth',1.5, 'MarkerSize',8);

% 1) FlatCE_Net：实线 + 红色 + 圆点标记
plot(p, FlatCE_Net, 'r-o','LineWidth',1.5, 'MarkerSize',8);

% 3) MMSE_NMSE：虚线 + 黑色 + 方块标记（最优解）
plot(p, MMSE_NMSE, 'k--s','LineWidth',1.5, 'MarkerSize',8);

xlabel('Pilot transmit power [dBm]','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('NMSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'LS','CDRN','FlatCE-Net','MMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

% legend({'FlatCE\_Net','MMSE\_NMSE','CDRN\_NMSE'},...
%        'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

grid on;  
set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
% title('Model Comparison at Different Power Levels');

% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/260525/15GHz_SNR_performance_comparison_270525_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SNR_performance_comparison_120225.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/260525/15GHz_SNR_performance_comparison_270525_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
