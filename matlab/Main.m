% clc;clear;
tic
K = 16; % Number of users
M = 4; % Number of antennas

L = 4; % Number of SIM layers
N = 32; % Number of elements per SIM layer

% sigma2 = db2pow(-126); % Noise power: -96 dBm (-126 dB)
sigma2 = 1; % Noise power: 0dB
frequency = 2*10^(9); % Carrier frequency
Pt_dB = -20; % Transmit power: 10 dBm (-20 dB)
Pt = db2pow(Pt_dB); % Transmit power
tau_p = K; % Pilot length

% F1: BS-SIM channel, shape: N,M 
% F: Intra-SIM channels, shape: N,N,L-1
[F1, F] = Channel_SIM(M, L, N, frequency); 

% Phase shift vectors
S = ceil(N/M); % Number of subframes 
ran_num = N*L*S; % Number of random phase shift vectors
phase = unifrnd(0,2*pi,N,L,S,ran_num); 
theta = exp(1i.*phase);

% Get the optimal V that minimizes the MSE of LS estimation, which
% is independent of system setup (user locations), V_LS shape: S*M, N
[V_LS, V_pinv_LS] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num);

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;
setup_num = 5000; % In each setup, user locations are randomly generated
realization_num = 100; % Number of channel realizations per setup

H_all = zeros(N,K,setup_num*realization_num);
H_est_LS_all = zeros(N,K,setup_num*realization_num);
% V_LS_all = zeros(S*M,N,setup_num*realization_num);
H_est_MMSE_all = zeros(N,K,setup_num*realization_num);
V_MMSE_all = zeros(S*M,N,setup_num*realization_num);
path_losses_all = zeros(K,1,setup_num*realization_num);

for setup = 1:setup_num
    % R_SIM: Correlation matrices including path loss, shape: N,N,K
    [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    % Get the optimal V that minimizes the MSE of MMSE estimation, 
    % which depends on system setup (user locations), V_MMSE shape: S*M, N
    [V_MMSE, RVPhi_inv] = Generate_GS_MMSE(theta, N,L,S,M,K, F, F1, ran_num, R_SIM, Pt, tau_p, sigma2);
    
    for realization = 1:realization_num
        % H: SIM-User channel, shape: N,K
        [H] = Channel_SIM_UE(K, N, R_SIM);
        H_norm = (norm(H, 'fro'))^2;
        H_norm_avg = H_norm_avg + H_norm/(setup_num*realization_num);

        [MSE_LS, H_est_LS] = Channel_estimation_LS(V_pinv_LS, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_LS_avg = MSE_LS_avg + MSE_LS/(setup_num*realization_num);
        
        [MSE_MMSE, H_est_MMSE] = Channel_estimation_MMSE(V_MMSE, RVPhi_inv, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_MMSE_avg = MSE_MMSE_avg + MSE_MMSE/(setup_num*realization_num);
        
        index = (setup-1)*realization_num + realization;
        fprintf("%d %d\n",setup,index);
        H_all(:,:,index) = H;
        H_est_LS_all(:,:,index) = H_est_LS;
        H_est_MMSE_all(:,:,index) = H_est_MMSE;
        V_MMSE_all(:,:,index) = V_MMSE;
        path_losses_all(:,:,index) = path_losses;
    end

end

NMSE_LS_avg = MSE_LS_avg/H_norm_avg
NMSE_MMSE_avg = MSE_MMSE_avg/H_norm_avg
% save('Dataset_10dBm_N36.mat', 'H_all', 'H_est_LS_all', 'H_est_MMSE_all', 'V_MMSE_all', 'path_losses_all');


H_all_data = H_all;
H_est_LS_all_data = H_est_LS_all;
H_est_MMSE_all_data = H_est_MMSE_all;


dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset5_withNMSE/';
data_type = 'train'
data_path = strcat(dir_path, data_type,'_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), '_Setup', num2str(setup_num), '_Reliz', num2str(realization_num), '.mat');
save(data_path, 'H_all_data', 'H_est_LS_all_data', 'H_est_MMSE_all_data', '-v7.3');


toc
