function [MSE, H_est] = Channel_estimation_MMSE(V_MMSE, RVPhi_inv, S, H, K, M, N, tau_p, Pt, sigma2)

H_est = zeros(N,K);
for k = 1:K
    hk = H(:,k);
    nk = sqrt(0.5*sigma2)*(randn(S*M,1)+1i*randn(S*M,1));
    yk = sqrt(Pt*tau_p)*V_MMSE*hk + nk;
    hk_est = sqrt(Pt*tau_p)*RVPhi_inv(:,:,k)*yk;
    H_est(:,k) = hk_est;
end

MSE = (norm(H - H_est, 'fro'))^2;

end