% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [5 10 15 20 25 ];


% --------------------latest fixed position---------------------------
LS_NMSE = [1.67262	0.50766	0.17983	0.04971	0.01558];
MMSE_NMSE = [0.17493	0.10181	0.05711	0.0272	0.01171];
FlatCE_Net = [0.192608	0.117309	0.059814	0.028213	0.011943];
CDRN_NMSE = [0.209734	0.129529	0.064356	0.030287	0.012892];

% --------------------15GHz fixed position---------------------------

LS_NMSE_15GHz = [1.85976, 0.58811, 0.17608, 0.06287, 0.01632];
MMSE_NMSE_15GHz = [0.18635, 0.10907, 0.0582, 0.02804, 0.01187];
FlatCE_Net_15GHz = [0.199116455, 0.116891099, 0.059325765, 0.032278498, 0.012053745];
CDRN_NMSE_15GHz = [0.20400721, 0.118668223, 0.06073253, 0.033121129, 0.013052685];



fig = figure('Visible', 'off');
hold on; box on;
% ---------------- Only markers for 2GHz ----------------
plot(p, LS_NMSE, 'b^', 'LineStyle','none','LineWidth',1.5, 'MarkerSize',8);
plot(p, CDRN_NMSE, 'md', 'LineStyle','none','LineWidth',1.5, 'MarkerSize',8);
plot(p, FlatCE_Net, 'ro', 'LineStyle','none','LineWidth',1.5, 'MarkerSize',8);
plot(p, MMSE_NMSE, 'ks', 'LineStyle','none','LineWidth',1.5, 'MarkerSize',8);

% ---------------- Lines only for 15GHz ----------------
plot(p, LS_NMSE_15GHz, 'b--','LineWidth',1.5);
plot(p, CDRN_NMSE_15GHz, 'm--', 'LineWidth',1.5);     % 15GHz CDRN (线)
plot(p, FlatCE_Net_15GHz, 'r-', 'LineWidth',1.5);    % 15GHz FlatCE (线)
plot(p, MMSE_NMSE_15GHz, 'k--', 'LineWidth',1.5);     % 15GHz MMSE (线)



xlabel('Pilot transmit power [dBm]','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('NMSE','Interpreter','Latex','Fontname', 'Times New Roman');

legend({...
    'LS@28GHz','CDRN@28GHz','FlatCE-Net@28GHz','MMSE@28GHz',...
    'LS@15GHz','CDRN@15GHz','FlatCE-Net@15GHz','MMSE@15GHz'},...
    'Interpreter','Latex', 'Fontname', 'Times New Roman', 'Location', 'Best');

% legend({...
%     'CDRN@28GHz','FlatCE-Net@28GHz','MMSE@28GHz',...
%     'CDRN@15GHz','FlatCE-Net@15GHz','MMSE@15GHz'},...
%     'Interpreter','Latex', 'Fontname', 'Times New Roman', 'Location', 'Best');

grid on;  
set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);
set(gca, 'YScale', 'log');
% title('Model Comparison at Different Power Levels');

% 保存图像到文件 (可以选择不同格式，如 PNG、EPS、PDF)
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/260525/2GHz_15GHz_SNR_performance_comparison_260525_log.png'); % 保存为 PNG 格式
% saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/SNR_performance_comparison_120225.pdf'); % 保存为 PDF 格式
saveas(fig, '/users/elq20xd/workspace/channel_estimation/flatCE_Net/figures/260525/2GHz_15GHz_SNR_performance_comparison_260525_log.eps', 'epsc'); % 保存为 EPS 格式
% print(fig, 'MSE_vs_TransmitPower', '-dpng', '-r300'); % 高分辨率保存 PNG

% 关闭图像窗口，避免占用内存
close(fig);
