% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [2,3,4,5,6];

% FlatCE_Net = [0.284101, 0.24624, 0.194411, 0.216062, 0.206377];
% LS_NMSE = [11.096421, 3.560983, 1.690374, 1.9375, 1.768164];
% MMSE_NMSE = [0.242319863, 0.220775224, 0.173299815, 0.186430324, 0.190794827];
% CDRN_NMSE = [0.216069, 0.187047, 0.06501, 0.04531, 0.016515];

FlatCE_Net = [0.284101, 0.24624, 0.194411, 0.216062, 0.206377];
LS_NMSE = [11.096421, 3.560983, 1.690374, 1.9375, 1.768164];
MMSE_NMSE = [0.242319863, 0.220775224, 0.173299815, 0.186430324, 0.190794827];
CDRN_NMSE = [0.32681,0.280037,0.216069,0.237657,0.245519];


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

xlabel('Number of layers','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('Normed MSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'FlatCE\_Net','LS\_NMSE','MMSE\_NMSE','CDRN\_NMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');


set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');


% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SIM_performance_comparison_120225_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SIM_performance_comparison_120225_log.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SIM_performance_comparison_120225_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
