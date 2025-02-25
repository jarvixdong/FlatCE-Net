% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [9,10,11,12,13];

% FlatCE_Net = [0.194411, 0.183997, 0.127541, 0.144586, 0.093255];
% LS_NMSE = [1.690374, 0.779491, 0.311853, 0.322463, 0.166494];
% MMSE_NMSE = [0.173299815, 0.178497301, 0.123810392, 0.137731364, 0.090993344];
% CDRN_NMSE = [0.216069,0.204332,0.146894,0.173717,0.108742];

% LS_NMSE = [1.64718, 0.4666, 0.27073, 0.17762, 0.13094];
% MMSE_NMSE = [0.18132, 0.13496, 0.10945, 0.09187, 0.07499];
% FlatCE_Net = [0.19899, 0.144737, 0.120778, 0.094537, 0.077308];
% CDRN_NMSE = [0.21476, 0.155506, 0.122884, 0.100652, 0.080864];

% LS_NMSE = [1.64718, 0.4666, 0.27073, 0.17762, 0.13094];
% MMSE_NMSE = [0.18132, 0.13496, 0.10945, 0.09187, 0.07499];
% FlatCE_Net = [0.19899, 0.144737, 0.115778, 0.094537, 0.077308];
% CDRN_NMSE = [0.21476, 0.155506, 0.122884, 0.100652, 0.080864];

% --------------------15dBm---------------------------
LS_NMSE = [0.1768, 0.04053, 0.03027, 0.02198, 0.0143];
MMSE_NMSE = [0.05564, 0.02736, 0.02351, 0.01812, 0.01242];
FlatCE_Net = [0.060751, 0.027917, 0.024103, 0.0183, 0.012562];
CDRN_NMSE = [0.066331, 0.029962, 0.025779, 0.019275, 0.012984];

% --------------------15dBm N4L4S9 Sutup10Rzl1000---------------------------
% LS_NMSE = [0.149892, 0.04053, 0.03027, 0.02198, 0.0143];
% MMSE_NMSE = [0.051635, 0.02736, 0.02351, 0.01812, 0.01242];
% FlatCE_Net = [0.054581, 0.027917, 0.024103, 0.0183, 0.012562];
% CDRN_NMSE = [0.06501, 0.029962, 0.025779, 0.019275, 0.012984];


fig = figure('Visible', 'off');
hold on; box on;
% plot(p, FlatCE_Net, 'b->','LineWidth',1.3, 'MarkerSize',8);
% plot(p, LS_NMSE, 'k-^','LineWidth',1.3, 'MarkerSize',8);
% plot(p, MMSE_NMSE, 'k--^','LineWidth',1.3, 'MarkerSize',8);
% plot(p, CDRN_NMSE, 'm-s','LineWidth',1.3, 'MarkerSize',8);

% 1) FlatCE_Net：实线 + 红色 + 圆点标记
plot(p, FlatCE_Net, 'r-o','LineWidth',1.5, 'MarkerSize',8);

% 2) LS_NMSE：虚线 + 蓝色 + 三角标记
plot(p, LS_NMSE, 'b--^','LineWidth',1.5, 'MarkerSize',8);

% 3) MMSE_NMSE：虚线 + 黑色 + 方块标记（最优解）
plot(p, MMSE_NMSE, 'k--s','LineWidth',1.5, 'MarkerSize',8);

% 4) CDRN_NMSE：虚线 + 洋红色 + 菱形标记
plot(p, CDRN_NMSE, 'm--d','LineWidth',1.5, 'MarkerSize',8);

% plot(p, averageMSE_Level3_with_TCO, 'm--s','LineWidth',1.3, 'MarkerSize',8);

xlabel('Length of subframe','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('Normed MSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'FlatCE\_Net','LS\_NMSE','MMSE\_NMSE','CDRN\_NMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

% legend({'FlatCE\_Net','MMSE\_NMSE','CDRN\_NMSE'},...
       % 'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');


set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
% title('Model Performance vs. Subframe Length at 15 dBm', 'FontSize', 14);
title('Performance vs. Subframe Length (15 dBm)');
set(gca, 'XMinorTick', 'off');



% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/250225/Subframe_performance_comparison_240225_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/180225/Subframe_performance_comparison_240225_log.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/250225/Subframe_performance_comparison_240225_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
