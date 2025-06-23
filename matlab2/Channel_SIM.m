function [F1, F] = Channel_SIM(M, L, N, frequency)

F1 = zeros(N,M); % BS-SIM channel, shape: N,M 
F = zeros(N,N,L-1); % Intra-SIM channels, shape: N,N,L-1
wave_len = (3*10^8)/frequency;
thick_SIM = 5*wave_len; % Thickness of SIM
dis_layer = thick_SIM/L; % Inter-layer distance
Nx = sqrt(N); % Number of elements on the x-axis
Ny = sqrt(N); % Number of elements on the y-axis
dx = wave_len/2; % Width per element
dy = wave_len/2; % Length per element

antenna_positions = zeros(M,1); % Antenna positions on the x-y plane
for m=1:M
    xm = (m-(M+1)/2)*(wave_len/2);
    antenna_positions(m) = xm + 1i*0; 
end

SIM_positions = zeros(N,1); % SIM positions on the x-y plane
for n=1:N
    xn = (mod(n-1,Nx) - (Nx-1)/2)*(wave_len/2);
    yn = (ceil(n/Nx) - (Ny+1)/2)*(wave_len/2);
    SIM_positions(n) = xn + 1i*yn;
end

F_any = zeros(N,N); % An intra-SIM channel (all intra-SIM channels are the same)
for n1=1:N
    for n2=1:N
        dis_xy = abs(SIM_positions(n1) - SIM_positions(n2));
        dis_n1n2 = sqrt(dis_xy^2 + dis_layer^2);
        cos_n1n2 = dis_layer/dis_n1n2;
        wn1n2 = dx*dy*cos_n1n2/dis_n1n2;
        wn1n2 = wn1n2*(1/(2*pi*dis_n1n2) - 1i/wave_len)*exp(1i*2*pi*dis_n1n2/wave_len);
        F_any(n2,n1) = wn1n2;
    end
end

% Incorrect version
% F_any = zeros(N,N); 
% for n1=1:N
%     for n2=1:N
%         dis_xy = sqrt((floor(abs(n1-n2)/Nx))^2 + (mod(abs(n1-n2),Nx))^2)*(wave_len/2);
%         dis_n1n2 = sqrt(dis_xy^2 + dis_layer^2);
%         cos_n1n2 = dis_layer/dis_n1n2;
%         wn1n2 = dx*dy*cos_n1n2/dis_n1n2;
%         wn1n2 = wn1n2*(1/(2*pi*dis_n1n2) - 1i/wave_len)*exp(1i*2*pi*dis_n1n2/wave_len);
%         F_any(n2,n1) = wn1n2;
%     end
% end

for l=1:L-1
    F(:,:,l) = F_any;
end

for m=1:M
    for n=1:N
        dis_xy = abs(antenna_positions(m) - SIM_positions(n));
        dis_mn = sqrt(dis_xy^2 + dis_layer^2);
        cos_mn = dis_layer/dis_mn;
        wmn = dx*dy*cos_mn/dis_mn;
        wmn = wmn*(1/(2*pi*dis_mn) - 1i/wave_len)*exp(1i*2*pi*dis_mn/wave_len);
        F1(n,m) = wmn;
    end
end

end