% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [8,9,10,11,12];

FlatCE_Net = [0.194411, 0.183997, 0.127541, 0.144586, 0.093255];
LS_NMSE = [1.690374, 0.779491, 0.311853, 0.322463, 0.166494];
MMSE_NMSE = [0.173299815, 0.178497301, 0.123810392, 0.137731364, 0.090993344];
CDRN_NMSE = [0.216069,0.204332,0.146894,0.173717,0.108742];


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


set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
% set(gca, 'YScale', 'log');


% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/Subframe_performance_comparison_120225.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/Subframe_performance_comparison_120225.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/Subframe_performance_comparison_120225.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
