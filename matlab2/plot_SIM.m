% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
% p = [2,3,4,5,6];
p = [2 4 6 8 10];

% FlatCE_Net = [0.284101, 0.24624, 0.194411, 0.216062, 0.206377];
% LS_NMSE = [11.096421, 3.560983, 1.690374, 1.9375, 1.768164];
% MMSE_NMSE = [0.242319863, 0.220775224, 0.173299815, 0.186430324, 0.190794827];
% CDRN_NMSE = [0.32681,0.280037,0.216069,0.237657,0.245519];

% LS_NMSE = [8.58414, 2.67756, 1.64718, 1.46813, 1.26637];
% MMSE_NMSE = [0.23361, 0.19511, 0.18132, 0.17629, 0.16834];
% FlatCE_Net = [0.265352, 0.205706, 0.19899, 0.195038, 0.202723];
% CDRN_NMSE = [0.30806, 0.224621, 0.21476, 0.209087, 0.21938];


% LS_NMSE = [8.58414, 2.67756, 1.64718, 1.46813, 1.26637];
% MMSE_NMSE = [0.23361, 0.19511, 0.18132, 0.17629, 0.16834];
% FlatCE_Net = [0.265352, 0.205706, 0.19899, 0.195038, 0.182723];
% CDRN_NMSE = [0.30806, 0.224621, 0.21476, 0.209087, 0.19938];

% --------------------15dBm---------------------------
% LS_NMSE = [0.79998, 0.23471, 0.1768, 0.14104, 0.11494];
% MMSE_NMSE = [0.10728, 0.06314, 0.05564, 0.04813, 0.04452];
% FlatCE_Net = [0.128572, 0.069338, 0.060751, 0.052476, 0.045855];
% CDRN_NMSE = [0.134438, 0.07678, 0.066331, 0.058008, 0.048997];

% % --------------------15dBm L5_S9_Setup20_Reliz2000---------------------------
% LS_NMSE = [0.79998, 0.23471, 0.1768, 0.13334, 0.11494];
% MMSE_NMSE = [0.10728, 0.06314, 0.05564, 0.05127, 0.04452];
% FlatCE_Net = [0.128572, 0.069338, 0.060751, 0.053864, 0.045855];
% CDRN_NMSE = [0.134438, 0.07678, 0.066331, 0.055192, 0.048997];


% --------------------15dBm N4L4S9 Sutup10Rzl1000---------------------------
% LS_NMSE = [0.79998, 0.23471, 0.149892, 0.14104, 0.11494];
% MMSE_NMSE = [0.10728, 0.06314, 0.051635, 0.04813, 0.04452];
% FlatCE_Net = [0.128572, 0.069338, 0.054581, 0.052476, 0.045855];
% CDRN_NMSE = [0.134438, 0.07678, 0.06501, 0.058008, 0.048997];

% --------------------fixed position 15dbm 23456---------------------------
% LS_NMSE = [0.84395 0.2387 0.17983 0.15424 0.13167];
% MMSE_NMSE = [0.09983 0.06486 0.05711 0.05289 0.05179];
% FlatCE_Net = [0.110641 0.073479 0.059814 0.064853 0.056019];
% CDRN_NMSE = [0.117038 0.080947 0.064356 0.070786 0.061473];


% --------------------fixed position 15dbm 246810---------------------------
LS_NMSE = [0.84395 0.17983 0.13167 0.11606 0.08076];
CDRN_NMSE = [0.117038 0.064356 0.061484 0.053492 0.045717];
FlatCE_Net = [0.110641 0.059814 0.056019 0.051415 0.041044];
MMSE_NMSE = [0.09983 0.05711 0.05179 0.04903 0.03908];
% LS_NMSE = [0.84395 0.17983 0.13167 0.11014 0.08076];
% CDRN_NMSE = [0.117038 0.064356 0.061484 0.054337 0.045717];
% FlatCE_Net = [0.110641 0.059814 0.056019 0.051613 0.041044];
% MMSE_NMSE = [0.09983 0.05711 0.05179 0.04603 0.03908];



fig = figure('Visible', 'off');
hold on; box on;
% plot(p, FlatCE_Net, 'b->','LineWidth',1.3, 'MarkerSize',8);
% plot(p, LS_NMSE, 'k-^','LineWidth',1.3, 'MarkerSize',8);
% plot(p, MMSE_NMSE, 'k--^','LineWidth',1.3, 'MarkerSize',8);
% plot(p, CDRN_NMSE, 'm-s','LineWidth',1.3, 'MarkerSize',8);

% 2) LS_NMSE：虚线 + 蓝色 + 三角标记
plot(p, LS_NMSE, 'b--^','LineWidth',1.5, 'MarkerSize',8);

% 4) CDRN_NMSE：虚线 + 洋红色 + 菱形标记
plot(p, CDRN_NMSE, 'm--d','LineWidth',1.5, 'MarkerSize',8);

% 1) FlatCE_Net：实线 + 红色 + 圆点标记
plot(p, FlatCE_Net, 'r-o','LineWidth',1.5, 'MarkerSize',8);

% 3) MMSE_NMSE：虚线 + 黑色 + 方块标记（最优解）
plot(p, MMSE_NMSE, 'k--s','LineWidth',1.5, 'MarkerSize',8);

% plot(p, averageMSE_Level3_with_TCO, 'm--s','LineWidth',1.3, 'MarkerSize',8);

xlabel('Number of metasurface layers','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('NMSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'LS','CDRN','FlatCE-Net','MMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

% legend({'FlatCE\_Net','MMSE\_NMSE','CDRN\_NMSE'},...
%        'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

grid on;  
set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
% title('Performance vs. SIM layer number (15 dBm)');


% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/150325/SIM_performance_comparison_210325_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SIM_performance_comparison_120225_log.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/150325/SIM_performance_comparison_210325_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
