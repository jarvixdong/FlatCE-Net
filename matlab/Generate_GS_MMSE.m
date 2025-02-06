function [V_opt, RVPhi_inv] = Generate_GS_MMSE(theta, N,L,S,M,K, F, F1, ran_num, R_SIM, Pt, tau_p, sigma2)

MSE_min = 10^15;
RVPhi_inv = zeros(N,S*M,K);

for i = 1:ran_num
    
    thetai = theta(:,:,:,i);

    % Diagonal matrices
    Phi = zeros(N,N,L,S);
    for s = 1:S
    for l = 1:L
        Phi(:,:,l,s) = diag(thetai(:,l,s));
    end
    end

    % Integrated intra-SIM channel, shape: N,N,S
    GS = zeros(N,N,S);
    for s = 1:S
    G = eye(N);
    for l = 1:L
        if l <= L-1
            G = F(:,:,l)*Phi(:,:,l,s)*G;
        else
            G = Phi(:,:,l,s)*G;
        end  
    end
    GS(:,:,s) = G;
    end

    V = zeros(N,S*M);

    for s = 1:S
        Gs = GS(:,:,s);
        Vs = F1'*Gs';
        V(:, M*(s-1)+1:M*s) = transpose(Vs);
    end

    V = transpose(V);
    
    MSE = 0;
    Phi_inv = zeros(S*M,S*M,K);
    for k = 1:K
        Phik_inv = (Pt*tau_p*V*R_SIM(:,:,k)*V' + sigma2*eye(S*M))^(-1);
        Phi_inv(:,:,k) = Phik_inv;
        MSE = MSE + trace(R_SIM(:,:,k) - Pt*tau_p*R_SIM(:,:,k)*V'*Phik_inv*V*R_SIM(:,:,k));
    end
    
    if MSE < MSE_min
        MSE_min = MSE;
        V_opt = V;
        for k = 1:K
            RVPhi_inv(:,:,k) = R_SIM(:,:,k)*V'*Phi_inv(:,:,k);
        end
    end

end


end