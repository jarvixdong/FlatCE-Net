% clc;clear;
tic
K = 4; % Number of users
M = 4; % Number of antennas 32,16,8,4,2

L = 4; % Number of SIM layers
N = 36; % Number of elements per SIM layer

% sigma2 = db2pow(-126); % Noise power: -96 dBm (-126 dB)
sigma2 = 1; % Noise power: 0dB
frequency = 2*10^(9); % Carrier frequency
Pt_dB = -5; % Transmit power: 10 dBm (-20 dB)
Pt = db2pow(Pt_dB); % Transmit power
tau_p = K; % Pilot length

% F1: BS-SIM channel, shape: N,M 
% F: Intra-SIM channels, shape: N,N,L-1
[F1, F] = Channel_SIM(M, L, N, frequency); 

% Phase shift vectors
S = 9; % Number of subframes S>ceil(N/M)
ran_num = N*L*S; % Number of random phase shift vectors
phase = unifrnd(0,2*pi,N,L,S,ran_num); 
theta = exp(1i.*phase);

% Get the optimal V that minimizes the MSE of LS estimation, which
% is independent of system setup (user locations), V_LS shape: S*M, N
[V_LS, V_pinv_LS] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num);

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;
setup_num = 10; % In each setup, user locations are randomly generated
realization_num = 1000; % Number of channel realizations per setup

H_all = zeros(N,K,setup_num*realization_num);
H_est_LS_all = zeros(N,K,setup_num*realization_num);
% V_LS_all = zeros(S*M,N,setup_num*realization_num);
H_est_MMSE_all = zeros(N,K,setup_num*realization_num);
% V_pinv_LS_all = zeros(S*M,N,setup_num*realization_num);
V_pinv_LS_all = zeros(N,S*M,setup_num*realization_num);

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
        fprintf("Pt_dB: %d L: %d M: %d S: %d Setup_num: %d Real_num: %d\n",Pt_dB,L,M,S,setup,index);
        H_all(:,:,index) = H;
        H_est_LS_all(:,:,index) = H_est_LS;
        H_est_MMSE_all(:,:,index) = H_est_MMSE;
        V_pinv_LS_all(:,:,index) = V_pinv_LS;
    end

end

NMSE_LS_avg = MSE_LS_avg/H_norm_avg
NMSE_MMSE_avg = MSE_MMSE_avg/H_norm_avg


valid_num = 10*realization_num;
H_all_train = H_all(:,:,1:end-valid_num);
H_est_LS_all_train = H_est_LS_all(:,:,1:end-valid_num);
H_est_MMSE_all_train = H_est_MMSE_all(:,:,1:end-valid_num);
V_pinv_LS_all_train = V_pinv_LS_all(:,:,1:end-valid_num);

H_all_test = H_all(:,:,end-valid_num+1:end);
H_est_LS_all_test = H_est_LS_all(:,:,end-valid_num+1:end);
H_est_MMSE_all_test = H_est_MMSE_all(:,:,end-valid_num+1:end);
V_pinv_LS_all_test = V_pinv_LS_all(:,:,end-valid_num+1:end);

dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset9_N36K4_v2/';
% train_data_path = strcat(dir_path, 'train','_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(setup_num), '_Reliz', num2str(realization_num),'_v1', '.mat')
% valid_data_path = strcat(dir_path, 'valid','_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(setup_num), '_Reliz', num2str(realization_num),'_v1', '.mat');

train_data_path = strcat(dir_path, 'train','_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(setup_num), '_Reliz', num2str(realization_num), '.mat')
valid_data_path = strcat(dir_path, 'valid','_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(setup_num), '_Reliz', num2str(realization_num), '.mat');

H_all_data = H_all_train;
H_est_LS_all_data = H_est_LS_all_train;
H_est_MMSE_all_data = H_est_MMSE_all_train;
V_pinv_LS_all_data = V_pinv_LS_all_train;
save(train_data_path, 'H_all_data', 'H_est_LS_all_data','H_est_MMSE_all_data',"V_pinv_LS_all_data",'-v7.3');

H_all_data = H_all_test;
H_est_LS_all_data = H_est_LS_all_test;
H_est_MMSE_all_data = H_est_MMSE_all_test;
V_pinv_LS_all_data = V_pinv_LS_all_test;
save(valid_data_path, 'H_all_data', 'H_est_LS_all_data','H_est_MMSE_all_data',"V_pinv_LS_all_data",'-v7.3');

toc
