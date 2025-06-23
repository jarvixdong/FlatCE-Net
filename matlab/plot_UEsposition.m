dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs/';

UEs_positions_path = strcat(dir_path, 'positionsUEs/UE_positions.mat');
UEspositions = load(UEs_positions_path);
train_UEs = UEspositions.UE_data.train;
valid_UEs = UEspositions.UE_data.valid;
test_UEs = UEspositions.UE_data.test;

% 创建一个新的图形窗口
figure;

% 绘制训练集 UE 位置
subplot(1, 3, 1);  % 1 行 3 列的子图，第 1 个
hold on;
for scene_id = 1:length(train_UEs)
    scatter(train_UEs(scene_id).UEx, train_UEs(scene_id).UEy, 'filled');
end
title('Training Set');
xlabel('X Position');
ylabel('Y Position');
grid on;
hold off;

% 绘制验证集 UE 位置
subplot(1, 3, 2);  % 1 行 3 列的子图，第 2 个
hold on;
for scene_id = 1:length(valid_UEs)
    scatter(valid_UEs(scene_id).UEx, valid_UEs(scene_id).UEy, 'filled');
end
title('Validation Set');
xlabel('X Position');
ylabel('Y Position');
grid on;
hold off;

% 绘制测试集 UE 位置
subplot(1, 3, 3);  % 1 行 3 列的子图，第 3 个
hold on;
for scene_id = 1:length(test_UEs)
    scatter(test_UEs(scene_id).UEx, test_UEs(scene_id).UEy, 'filled');
end
title('Test Set');
xlabel('X Position');
ylabel('Y Position');
grid on;
hold off;

% 调整子图间距
sgtitle('UE Positions in Training, Validation, and Test Sets');

% 保存图形
save_path = strcat(dir_path, 'positionsUEs/UE_positions_plot.png');  % 保存路径
saveas(gcf, save_path);  % 保存当前图形窗口
disp(['UE positions plot saved to: ', save_path]);