function [V_opt, V_pinv] = Generate_GS_LS(theta, N,L,S,M, F, F1, ran_num)

MSE_min = 10^15;

for i = 1: ran_num
    
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
    
    VV = V'*V;
    VV_inv = VV^(-1);
    
    if trace(VV_inv) < MSE_min
        MSE_min = trace(VV_inv);
        V_opt = V;
        V_pinv = VV_inv*V';
    end

end


end