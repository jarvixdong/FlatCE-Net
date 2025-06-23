% 设置参数
num_train = 100;  % 训练集场景数
num_valid = 10;   % 验证集场景数
num_test = 10;    % 测试集场景数
K = 4;            % 每个场景的 UE 数量

% 目标存储文件
save_file = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs_v5/positionsUEs/UE_positions.mat';

% 初始化 UE_data 结构体
UE_data = struct();  % 初始化 UE_data 为一个标量结构体

% 生成训练集
UE_data.train = struct('UEx', cell(1, num_train), 'UEy', cell(1, num_train));  % 预分配内存
for scene_id = 1:num_train
    UEx = 10 + rand(1, K) * 20;
    UEy = -10 + rand(1, K) * 20;
    UE_data.train(scene_id).UEx = UEx;
    UE_data.train(scene_id).UEy = UEy;
end

% 生成验证集
UE_data.valid = struct('UEx', cell(1, num_valid), 'UEy', cell(1, num_valid));  % 预分配内存
for scene_id = 1:num_valid
    UEx = 10 + rand(1, K) * 20;
    UEy = -10 + rand(1, K) * 20;
    UE_data.valid(scene_id).UEx = UEx;
    UE_data.valid(scene_id).UEy = UEy;
end

% 生成测试集
UE_data.test = struct('UEx', cell(1, num_test), 'UEy', cell(1, num_test));  % 预分配内存
for scene_id = 1:num_test
    UEx = 10 + rand(1, K) * 20;
    UEy = -10 + rand(1, K) * 20;
    UE_data.test(scene_id).UEx = UEx;
    UE_data.test(scene_id).UEy = UEy;
end

% 保存到 .mat 文件
save(save_file, 'UE_data');

disp(['All UE positions saved in: ', save_file]);