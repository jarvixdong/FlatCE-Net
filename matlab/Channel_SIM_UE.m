function [H] = Channel_SIM_UE(K, N, R_SIM)

% Generate uncorrelated Rayleigh fading channel realizations
H = sqrt(0.5)*(randn(N,K)+1i*randn(N,K));

for k = 1:K
    % Rsqrt: N x N
    Rsqrt = sqrtm(R_SIM(:,:,k)); 
    H(:,k) = Rsqrt*H(:,k);
end


end