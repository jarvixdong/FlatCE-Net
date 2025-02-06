import numpy as np
import h5py
from scipy.io import loadmat


def load_data(path):
    data = loadmat(path)
    print(data.keys())
    H_all = data["H_all"].transpose(2,0,1)
    H_est_LS_all = data["H_est_LS_all"].transpose(2,0,1)
            
    H_est_LS_all_unfold = np.concatenate([H_est_LS_all.real[:,np.newaxis,:,:],H_est_LS_all.imag[:,np.newaxis,:,:]],axis=1)
    H_all_unfold = np.concatenate([H_all.real[:,np.newaxis,:,:],H_all.imag[:,np.newaxis,:,:]],axis=1)


    H_all_unfold_sum = H_all_unfold.sum(axis=(1,2,3))
    H_all_unfold = H_all_unfold[H_all_unfold_sum!=0]
    H_est_LS_all_unfold = H_est_LS_all_unfold[H_all_unfold_sum!=0]

    print("H_all:",H_all_unfold.shape,H_all_unfold[0][0][:,0])
    print("H_est_LS_all_unfold:",H_est_LS_all_unfold.shape,H_est_LS_all_unfold[0][0][:,0])
    print("nmse:",cal_NMSE2(H_est_LS_all_unfold,H_all_unfold))

    return H_all_unfold,H_est_LS_all_unfold


def cal_NMSE2(H_noise, H_clean):
    # 计算误差和信号的模平方
    # print("H clean:",H_clean[0][0][:,0])
    H_dis = (H_noise - H_clean) ** 2
    H_clean_sq = H_clean ** 2
    # print('H dis:',H_dis.shape)
    # print('H_clean_sq:',H_clean_sq.shape)

    H_dis_sum = H_dis.sum(axis=( 1, 2, 3))
    H_clean_sq_sum = H_clean_sq.sum(axis=( 1, 2, 3))
    # print('H_dis_sum dis:',H_dis_sum.shape,H_dis_sum[0],H_clean_sq_sum[0])
    
    nmmse_batch = H_dis_sum/H_clean_sq_sum

    nmmse_mean = nmmse_batch.mean()
    
    return nmmse_mean

    
def cal_NMMSE(H_clean, H_denoise):
    # 计算误差和信号的模平方
    H_dis = np.abs(H_clean - H_denoise) ** 2
    H_clean_abssq = np.abs(H_clean) ** 2
    # print(H_clean_abssq)

    # 求和，减少多次 sum 调用
    # H_dis_sum = H_dis.sum(axis=(0, 1, 2))
    # H_clean_abssq_sum = H_clean_abssq.sum(axis=(0, 1, 2))

    # H_dis_sum = H_dis.sum(axis=( 1, 2))
    # H_clean_abssq_sum = H_clean_abssq.sum(axis=( 1, 2))
    
    H_dis_sum = H_dis.mean(axis=( 1, 2))
    H_clean_abssq_sum = H_clean_abssq.mean(axis=( 1, 2))
    
    # print("H_clean_abssq_sum:",H_clean_abssq_sum)
    # print('H_clean_abssq',H_clean_abssq[-2])
    # print("H_clean:",H_clean[-2])
    
    # 避免除零错误
    nmmse = np.where(H_clean_abssq_sum > 0, H_dis_sum / H_clean_abssq_sum, float('inf'))

    # 计算平均 NMMSE
    # nmmse_mean = nmmse.mean()
    nmmse_mean = np.mean(nmmse[~np.isinf(nmmse)])

    # 输出结果
    print('nmmse:', nmmse)
    print('nmmse_mean:', nmmse_mean)
    

    return nmmse_mean, nmmse
    

def trans_complex2ptheta(complex_num,ptheta):
    complex_mat = np.concatenate([complex_num.real[np.newaxis,:,:,:],complex_num.imag[np.newaxis,:,:,:]],axis=0).transpose(3,0,1,2)
        
        
        
# path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/valid_Dataset_dB-20_N36_K4_Setup5_Reliz100.mat'
path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset2/valid_Dataset_dB10_N32_K16_Setup5_Reliz100.mat'
# path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/train_Dataset_dB-20_N36_K4_Setup5_Reliz10000.mat'
H_all,H_est_LS_all = load_data(path)
H_all = H_all[0:32]
H_est_LS_all = H_est_LS_all[0:32]
nmse = cal_NMSE2(H_est_LS_all,H_all)
print('nmse:',nmse)
