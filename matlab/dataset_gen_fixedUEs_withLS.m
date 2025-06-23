% clc;clear;
tic
K = 4; % Number of users
M = 4; % Number of antennas 32,16,8,4,2

L = 4; % Number of SIM layers
N = 36; % Number of elements per SIM layer

% sigma2 = db2pow(-126); % Noise power: -96 dBm (-126 dB)
sigma2 = 1; % Noise power: 0dB
frequency = 2*10^(9); % Carrier frequency
Pt_dB = -15; % Transmit power: 10 dBm (-20 dB)
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
train_setup_num = 200; % In each setup, user locations are randomly generated
valid_setup_num = 20; % In each setup, user locations are randomly generated
test_setup_num = 20; % In each setup, user locations are randomly generated
% realization_num = 10000; % Number of channel realizations per setup
train_reliz = 1000
valid_reliz = 1000
test_reliz = 1000

% -----------------------------------train dataset part----------------------------
H_all_train = zeros(N,K,train_setup_num*train_reliz);
H_est_LS_all_train = zeros(N,K,train_setup_num*train_reliz);
H_est_MMSE_all_train = zeros(N,K,train_setup_num*train_reliz);
V_LS_all_train = zeros(S*M,N,train_setup_num*train_reliz);

dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs/';
% dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs_v3/';
% dir_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs_v2/';

UEs_positions_path = strcat(dir_path,'positionsUEs/UE_positions.mat')
UEspositions = load(UEs_positions_path);
train_UEs = UEspositions.UE_data.train;
valid_UEs =  UEspositions.UE_data.valid;
test_UEs =  UEspositions.UE_data.test;


for setup = 1:train_setup_num
    % [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    [R_SIM, path_losses] = Channel_statistics_fixedUEs(K, N, frequency, train_UEs(setup));
    [V_MMSE, RVPhi_inv] = Generate_GS_MMSE(theta, N,L,S,M,K, F, F1, ran_num, R_SIM, Pt, tau_p, sigma2);
    
    for realization = 1:train_reliz
        [H] = Channel_SIM_UE(K, N, R_SIM);
        H_norm = (norm(H, 'fro'))^2;
        H_norm_avg = H_norm_avg + H_norm/(train_setup_num*train_reliz);

        [MSE_LS, H_est_LS] = Channel_estimation_LS(V_pinv_LS, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_LS_avg = MSE_LS_avg + MSE_LS/(train_setup_num*train_reliz);
        
        [MSE_MMSE, H_est_MMSE] = Channel_estimation_MMSE(V_MMSE, RVPhi_inv, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_MMSE_avg = MSE_MMSE_avg + MSE_MMSE/(train_setup_num*train_reliz);
        
        index = (setup-1)*train_reliz + realization;
        fprintf("Pt_dB: %d L: %d M: %d S: %d Setup_num: %d Real_num: %d\n",Pt_dB,L,M,S,setup,index);
        H_all_train(:,:,index) = H;
        H_est_LS_all_train(:,:,index) = H_est_LS;
        H_est_MMSE_all_train(:,:,index) = H_est_MMSE;
        V_LS_all_train(:,:,index) = V_LS;
    end

end

train_NMSE_LS_avg = MSE_LS_avg/H_norm_avg;
train_NMSE_MMSE_avg = MSE_MMSE_avg/H_norm_avg;

% -----------------------------------valid dataset part----------------------------

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;

H_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
H_est_LS_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
H_est_MMSE_all_valid = zeros(N,K,valid_setup_num*valid_reliz);
V_LS_all_valid = zeros(S*M,N,valid_setup_num*valid_reliz);

for setup = 1:valid_setup_num
    % [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    [R_SIM, path_losses] = Channel_statistics_fixedUEs(K, N, frequency, valid_UEs(setup));
    [V_MMSE, RVPhi_inv] = Generate_GS_MMSE(theta, N,L,S,M,K, F, F1, ran_num, R_SIM, Pt, tau_p, sigma2);
    
    for realization = 1:valid_reliz
        [H] = Channel_SIM_UE(K, N, R_SIM);
        H_norm = (norm(H, 'fro'))^2;
        H_norm_avg = H_norm_avg + H_norm/(valid_setup_num*valid_reliz);

        [MSE_LS, H_est_LS] = Channel_estimation_LS(V_pinv_LS, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_LS_avg = MSE_LS_avg + MSE_LS/(valid_setup_num*valid_reliz);
        
        [MSE_MMSE, H_est_MMSE] = Channel_estimation_MMSE(V_MMSE, RVPhi_inv, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_MMSE_avg = MSE_MMSE_avg + MSE_MMSE/(valid_setup_num*valid_reliz);
        
        index = (setup-1)*valid_reliz + realization;
        fprintf("Pt_dB: %d L: %d M: %d S: %d Setup_num: %d Real_num: %d\n",Pt_dB,L,M,S,setup,index);
        H_all_valid(:,:,index) = H;
        H_est_LS_all_valid(:,:,index) = H_est_LS;
        H_est_MMSE_all_valid(:,:,index) = H_est_MMSE;
        V_LS_all_valid(:,:,index) = V_LS;
    end

end

valid_NMSE_LS_avg = MSE_LS_avg/H_norm_avg;
valid_NMSE_MMSE_avg = MSE_MMSE_avg/H_norm_avg;

% -----------------------------------test dataset part----------------------------

MSE_LS_avg = 0;
MSE_MMSE_avg = 0;
H_norm_avg = 0;

H_all_test = zeros(N,K,test_setup_num*test_reliz);
H_est_LS_all_test = zeros(N,K,test_setup_num*test_reliz);
H_est_MMSE_all_test = zeros(N,K,test_setup_num*test_reliz);
V_LS_all_test = zeros(S*M,N,test_setup_num*test_reliz);

for setup = 1:test_setup_num
    % [R_SIM, path_losses] = Channel_statistics(K, N, frequency);
    [R_SIM, path_losses] = Channel_statistics_fixedUEs(K, N, frequency, test_UEs(setup));
    [V_MMSE, RVPhi_inv] = Generate_GS_MMSE(theta, N,L,S,M,K, F, F1, ran_num, R_SIM, Pt, tau_p, sigma2);
    
    for realization = 1:test_reliz
        [H] = Channel_SIM_UE(K, N, R_SIM);
        H_norm = (norm(H, 'fro'))^2;
        H_norm_avg = H_norm_avg + H_norm/(test_setup_num*test_reliz);

        [MSE_LS, H_est_LS] = Channel_estimation_LS(V_pinv_LS, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_LS_avg = MSE_LS_avg + MSE_LS/(test_setup_num*test_reliz);
        
        [MSE_MMSE, H_est_MMSE] = Channel_estimation_MMSE(V_MMSE, RVPhi_inv, S, H, K, M, N, tau_p, Pt, sigma2);
        MSE_MMSE_avg = MSE_MMSE_avg + MSE_MMSE/(test_setup_num*test_reliz);
        
        index = (setup-1)*test_reliz + realization;
        fprintf("Pt_dB: %d L: %d M: %d S: %d Setup_num: %d Real_num: %d\n",Pt_dB,L,M,S,setup,index);
        H_all_test(:,:,index) = H;
        H_est_LS_all_test(:,:,index) = H_est_LS;
        H_est_MMSE_all_test(:,:,index) = H_est_MMSE;
        V_LS_all_test(:,:,index) = V_LS;
    end

end

test_NMSE_LS_avg = MSE_LS_avg/H_norm_avg;
test_NMSE_MMSE_avg = MSE_MMSE_avg/H_norm_avg;

% -----------------------------------save dataset part----------------------------


train_suffix = strcat('train_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), ...
                      '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(train_setup_num), ...
                      '_Reliz', num2str(train_reliz), 'withLS.mat');
train_data_path = strcat(dir_path, train_suffix);

valid_suffix = strcat('valid_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), ...
                      '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(valid_setup_num), ...
                      '_Reliz', num2str(valid_reliz), 'withLS.mat');
valid_data_path = strcat(dir_path, valid_suffix);

test_suffix = strcat('test_Dataset_dB', num2str(Pt_dB), '_N', num2str(N), '_K', num2str(K), ...
                     '_L', num2str(L), '_S', num2str(S), '_Setup', num2str(test_setup_num), ...
                     '_Reliz', num2str(test_reliz), 'withLS.mat');
test_data_path = strcat(dir_path, test_suffix);


H_all_data = H_all_train;
H_est_LS_all_data = H_est_LS_all_train;
H_est_MMSE_all_data = H_est_MMSE_all_train;
V_LS_all_data = V_LS_all_train;
save(train_data_path, 'H_all_data', 'H_est_LS_all_data','H_est_MMSE_all_data',"V_LS_all_data",'-v7.3');

H_all_data = H_all_valid;
H_est_LS_all_data = H_est_LS_all_valid;
H_est_MMSE_all_data = H_est_MMSE_all_valid;
V_LS_all_data = V_LS_all_valid;
save(valid_data_path, 'H_all_data', 'H_est_LS_all_data','H_est_MMSE_all_data',"V_LS_all_data",'-v7.3');

H_all_data = H_all_test;
H_est_LS_all_data = H_est_LS_all_test;
H_est_MMSE_all_data = H_est_MMSE_all_test;
V_LS_all_data = V_LS_all_test;
save(test_data_path, 'H_all_data', 'H_est_LS_all_data','H_est_MMSE_all_data',"V_LS_all_data",'-v7.3');


fprintf("train_NMSE_LS_avg: %.5f MMSE_avg: %.5f \n", train_NMSE_LS_avg, train_NMSE_MMSE_avg);
fprintf("valid_NMSE_LS_avg: %.5f MMSE_avg: %.5f \n", valid_NMSE_LS_avg, valid_NMSE_MMSE_avg);
fprintf("test_NMSE_LS_avg: %.5f MMSE_avg: %.5f \n", test_NMSE_LS_avg, test_NMSE_MMSE_avg);
% fprintf("%d,%d,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",N,K,test_setup_num,realization_num,Pt_dB,L,M,S,train_NMSE_LS_avg, train_NMSE_MMSE_avg,valid_NMSE_LS_avg, valid_NMSE_MMSE_avg,test_NMSE_LS_avg, test_NMSE_MMSE_avg)
fprintf("%d,%d,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s\n", ...
    N, K, test_setup_num, train_reliz, Pt_dB, L, M, S, ...
    train_NMSE_LS_avg, train_NMSE_MMSE_avg, valid_NMSE_LS_avg, valid_NMSE_MMSE_avg, ...
    test_NMSE_LS_avg, test_NMSE_MMSE_avg, train_suffix, valid_suffix, test_suffix);

toc
