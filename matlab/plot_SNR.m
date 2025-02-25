% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [5 10 15 20 25 ];

% FlatCE_Net = [0.194411, 0.15893, 0.054581, 0.036471, 0.013426];
% LS_NMSE = [1.690374, 1.004105, 0.149892, 0.062682, 0.01751];
% MMSE_NMSE = [0.173299815, 0.137608864, 0.051634597, 0.030899951, 0.012977496];
% CDRN_NMSE = [0.216069, 0.187047, 0.06501, 0.04531, 0.016515];

% --------------------15dBm N4L4S9 is Sutup20Rzl1000---------------------------
LS_NMSE = [1.64718, 0.4829, 0.17019, 0.04062, 0.01275];
MMSE_NMSE = [0.18132, 0.10282, 0.0524, 0.02446, 0.01009];
FlatCE_Net = [0.19899, 0.118893, 0.059984, 0.025657, 0.010624];
CDRN_NMSE = [0.21476, 0.149353, 0.076029, 0.030888, 0.011644];

% --------------------15dBm N4L4S9 is Sutup10Rzl1000---------------------------
% LS_NMSE = [1.64718, 0.4829, 0.149892, 0.04062, 0.01275];
% MMSE_NMSE = [0.18132, 0.10282, 0.0516345, 0.02446, 0.01009];
% FlatCE_Net = [0.19899, 0.118893, 0.054581, 0.025657, 0.010624];
% CDRN_NMSE = [0.21476, 0.149353, 0.06501, 0.030888, 0.011644];


fig = figure('Visible', 'off');
hold on; box on;
% 1) FlatCE_Net：实线 + 红色 + 圆点标记
plot(p, FlatCE_Net, 'r-o','LineWidth',1.5, 'MarkerSize',8);

% 2) LS_NMSE：虚线 + 蓝色 + 三角标记
plot(p, LS_NMSE, 'b--^','LineWidth',1.5, 'MarkerSize',8);

% 3) MMSE_NMSE：虚线 + 黑色 + 方块标记（最优解）
plot(p, MMSE_NMSE, 'k--s','LineWidth',1.5, 'MarkerSize',8);

% 4) CDRN_NMSE：虚线 + 洋红色 + 菱形标记
plot(p, CDRN_NMSE, 'm--d','LineWidth',1.5, 'MarkerSize',8);
% plot(p, averageMSE_Level3_with_TCO, 'm--s','LineWidth',1.3, 'MarkerSize',8);

xlabel('Transmit power [dBm]','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('Normed MSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'FlatCE\_Net','LS\_NMSE','MMSE\_NMSE','CDRN\_NMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

% legend({'FlatCE\_Net','MMSE\_NMSE','CDRN\_NMSE'},...
%        'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');


set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
title('Model Comparison at Different Power Levels');

% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/250225/SNR_performance_comparison_240225_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SNR_performance_comparison_120225.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/250225/SNR_performance_comparison_240225_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
