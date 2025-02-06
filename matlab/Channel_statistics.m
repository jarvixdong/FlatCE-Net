function [R_SIM, path_losses] = Channel_statistics(K, N, frequency)

wave_len = (3*10^8)/frequency; % Wavelength
thick_SIM = 5*wave_len; % Thickness of SIM
Nx = sqrt(N); % Number of elements on the x-axis
height_BS = 10; % BS height
height_difference = height_BS - thick_SIM; 
path_losses = zeros(K,1);
sigma2 = db2pow(-126); % Noise power: -96 dBm (-126 dB)
C0 = 10^(-3.05)/sigma2; % !!!!!!

for k = 1:K
%     UEx = 10 + (k-1)*10;
%     UEy = 0;
    UEx = 10 + rand(1,1)*20;
    UEy = -10 + rand(1,1)*20;
    UEposition = UEx + 1i*UEy;
    dis_xy = abs(UEposition);

    distance = sqrt(height_difference^2 + dis_xy^2);
    path_loss = C0*distance^(-3.67);
    path_losses(k) = path_loss;
end

R = eye(N);  % Correlation matrix for small-scale fading
for i = 1:N
    for j = 1:N
        x_i = mod(i-1, Nx);
        y_i = floor((i-1)/Nx);
        x_j = mod(j-1, Nx);
        y_j = floor((j-1)/Nx);
        dis_ij = ((x_i-x_j)^2 + (y_i-y_j)^2)^0.5;
        R(i, j) = sinc(dis_ij);
    end
end

R_SIM = zeros(N,N,K); % Correlation matrices including path loss
for k = 1:K
    R_SIM(:,:,k) = path_losses(k)*R;
end


end