% clc;clear;
tic
K = 4; % Number of users
M = 4; % Number of antennas 32,16,8,4,2

L = 6; % Number of SIM layers
N = 36; % Number of elements per SIM layer

% sigma2 = db2pow(-126); % Noise power: -96 dBm (-126 dB)
sigma2 = 1; % Noise power: 0dB
frequency = 15*10^(9); % Carrier frequency
Pt_dB = -15; % Transmit power: 10 dBm (-20 dB)
Pt = db2pow(Pt_dB); % Transmit power
tau_p = K; % Pilot length

% F1: BS-SIM channel, shape: N,M 
% F: Intra-SIM channels, shape: N,N,L-1
[F1, F] = Channel_SIM(M, L, N, frequency); 

% Phase shift vectors
S = 13; % Number of subframes S>ceil(N/M)
ran_num = N*L*S % Number of random phase shift vectors
phase = unifrnd(0,2*pi,N,L,S,ran_num); 
theta = exp(1i.*phase);

% % Get the optimal V that minimizes the MSE of LS estimation, which
% % is independent of system setup (user locations), V_LS shape: S*M, N
% [V_LS, V_pinv_LS] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num);

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;
train_setup_num = 0; % In each setup, user locations are randomly generated
valid_setup_num = 100; % In each setup, user locations are randomly generated
test_setup_num = 0; % In each setup, user locations are randomly generated
% realization_num = 10000; % Number of channel realizations per setup

valid_reliz = 10000
dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs/';


UEs_positions_path = strcat(dir_path,'positionsUEs/UE_positions.mat')
UEspositions = load(UEs_positions_path);
valid_UEs =  UEspositions.UE_data.valid;


% -----------------------------------valid dataset part----------------------------

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;

H_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
H_est_LS_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
H_est_MMSE_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
V_pinv_LS_all_valid = zeros(N,S*M,valid_setup_num*valid_reliz);

[V_LS, V_pinv_LS] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num);

tic;  % 开始计时
% Get the optimal V that minimizes the MSE of LS estimation, which
% is independent of system setup (user locations), V_LS shape: S*M, N

% [V_LS, V_pinv_LS] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num);

for setup = 1:valid_setup_num
    fprintf("setup %d %d \n", setup,valid_setup_num);
    [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    for realization = 1:valid_reliz
        [H] = Channel_SIM_UE(K, N, R_SIM);
        [MSE_LS, H_est_LS] = Channel_estimation_LS(V_pinv_LS, S, H, K, M, N, tau_p, Pt, sigma2);
    end

end
elapsed_time = toc;  % 输出用时
fprintf("whole time in %.4f seconds.\n", elapsed_time);


tic;  % 开始计时
for setup = 1:valid_setup_num
    [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    for realization = 1:valid_reliz
        [H] = Channel_SIM_UE(K, N, R_SIM);
    end
end
elapsed_time = toc;  % 输出用时
fprintf("Generate H in %.4f seconds.\n", elapsed_time);





