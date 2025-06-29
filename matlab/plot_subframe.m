% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [9,10,11,12,13];


% % --------------------15dBm---------------------------
% LS_NMSE = [0.1768, 0.04053, 0.03027, 0.02198, 0.0143];
% MMSE_NMSE = [0.05564, 0.02736, 0.02351, 0.01812, 0.01242];
% FlatCE_Net = [0.060751, 0.027917, 0.024103, 0.0183, 0.012562];
% CDRN_NMSE = [0.066331, 0.029962, 0.025779, 0.019275, 0.012984];

% % --------------------fixed position 15dbm---------------------------
% LS_NMSE = [0.17983	0.05569	0.02957	0.01933	0.01399];
% MMSE_NMSE = [0.05711	0.03194	0.02128	0.01601	0.01208];
% FlatCE_Net = [0.059814	0.034383	0.021596	0.016224	0.012218];
% CDRN_NMSE = [0.064356	0.037902	0.023121	0.017148	0.012747];


% --------------------fixed position 15dbm 15GHz---------------------------
LS_NMSE = [0.17608	0.05329	0.03173	0.02027	0.01519];
MMSE_NMSE = [0.0582	0.03477	0.02411	0.01693	0.01309];
FlatCE_Net = [0.059325765	0.035227714	0.024683874	0.01704095	0.013193002];
CDRN_NMSE = [0.06073253	0.036624096	0.026327504	0.017946553	0.013950874];
attn_NMSE = [0.065538961	0.042457968	0.028064399	0.01854342	0.014158169];


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

% 5) NewMethod_NMSE：虚线 + 绿色 + 星形标记
plot(p, attn_NMSE, 'g-->','LineWidth',1.5, 'MarkerSize',8);

% 1) FlatCE_Net：实线 + 红色 + 圆点标记
plot(p, FlatCE_Net, 'r-o','LineWidth',1.5, 'MarkerSize',8);

% 3) MMSE_NMSE：虚线 + 黑色 + 方块标记（最优解）
plot(p, MMSE_NMSE, 'k--s','LineWidth',1.5, 'MarkerSize',8);


% plot(p, averageMSE_Level3_with_TCO, 'm--s','LineWidth',1.3, 'MarkerSize',8);

xlabel('Number of pilot subframes','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('NMSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({'LS','CDRN','FCEM','FlatCE-Net','MMSE'},...
       'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

% legend({'FlatCE\_Net','MMSE\_NMSE','CDRN\_NMSE'},...
       % 'Interpreter','Latex','Fontname', 'Times New Roman', 'Location', 'Best');

grid on;  
set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
% title('Model Performance vs. Subframe Length at 15 dBm', 'FontSize', 14);
% title('Performance vs. Subframe Length (15 dBm)');
set(gca, 'XMinorTick', 'off');



% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/290625/15GHz_Subframe_performance_comparison_290625_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/180225/Subframe_performance_comparison_240225_log.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/290625/15GHz_Subframe_performance_comparison_290625_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
