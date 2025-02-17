% Plot MSE versus the transmit power
% L = 64; N = 4; K = 10; nbrBSs = 4; nbrgroups = 1; tau_p = 10; p_pilot = 100 mW;

% Distribution 1: users in group g uniformly ditrsiued in cell g

%Uplink transmit power per UE (dBm)
p = [10 15 20 25 30 35 40];

averageMSE_Level1 = [-33.2998  -33.8886  -34.3541  -34.6525  -34.8081  -34.8756  -34.9010];
averageMSE_Cellular_no_TCO = [-46.1516  -50.3410  -53.9736  -56.4185  -57.6352  -58.1102  -58.2729];
averageMSE_Cellular_with_TCO = [-47.2847  -51.1060  -54.5709  -57.1724  -58.6324  -59.2580  -59.4827];
averageMSE_Level3_no_TCO = [-60.4512  -64.8022  -68.2467  -70.3726  -71.3611  -71.7338  -71.8598];
averageMSE_Level3_with_TCO = [-60.5588  -64.8433  -68.2729  -70.4029  -71.3978  -71.7738  -71.9011];

figure;
hold on; box on;
plot(p, averageMSE_Level1, 'b->','LineWidth',1.3, 'MarkerSize',8);
plot(p, averageMSE_Cellular_no_TCO, 'k-^','LineWidth',1.3, 'MarkerSize',8);
plot(p, averageMSE_Cellular_with_TCO, 'k--^','LineWidth',1.3, 'MarkerSize',8);
plot(p, averageMSE_Level3_no_TCO, 'm-s','LineWidth',1.3, 'MarkerSize',8);
plot(p, averageMSE_Level3_with_TCO, 'm--s','LineWidth',1.3, 'MarkerSize',8);
xlabel('Maximum transmit power per device [dBm]','Interpreter','Latex','Fontname', 'Times New Roman');
ylabel('MSE [dB]','Interpreter','Latex','Fontname', 'Times New Roman');
legend({'Level 1','Cellular mMIMO w/o TCO','Cellular mMIMO w/ TCO','Level 2/3 w/o TCO','Level 2/3 w/ TCO'},'Interpreter','Latex','Fontname', 'Times New Roman');
% ylim([-32, -10]);
set(gca,'FontSize',18);
set(gca, 'LooseInset', [0,0,0,0]);