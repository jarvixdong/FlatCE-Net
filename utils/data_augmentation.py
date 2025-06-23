import numpy as np
import h5py
import torch


def shuffle_dim1_numpy_batch(x,y):
    batch_size, num_channels, seq_len = x.shape  # (200000, 4, 36)

    # 生成每个样本的随机通道索引 (200000, 4)
    indices = np.apply_along_axis(np.random.permutation, 1, np.tile(np.arange(num_channels), (batch_size, 1)))

    # 使用高级索引进行通道重排
    shuffled_x = x[np.arange(batch_size)[:, None], indices, :]
    shuffled_y = y[np.arange(batch_size)[:, None], indices, :]

    return shuffled_x,shuffled_y


def data_preAugment(src):
    data = h5py.File(src, 'r') 
    V_complex = data['V_pinv_LS_all_data'][:]
    
    xdata_complex = data['H_est_LS_all_data'][:]
    ydata_complex = data['H_all_data'][:]
    print('data shape:',V_complex.shape,xdata_complex.shape,ydata_complex.shape)
    shuffled_x,shuffled_y = shuffle_dim1_numpy_batch(xdata_complex,ydata_complex)
    print("shufed x,y",xdata_complex.shape,ydata_complex.shape)
    # print("x0",xdata_complex[0,:,0:5],shuffled_x[0,:,0:5])
    # print('y0',ydata_complex[0,:,0:5],shuffled_y[0,:,0:5])
    data_saved_x = np.concatenate([xdata_complex,shuffled_x],axis=0)
    data_saved_y = np.concatenate([ydata_complex,shuffled_y],axis=0)
    data_saved_V = np.concatenate([V_complex,V_complex],axis=0)
    print('data saved',data_saved_x.shape,data_saved_y.shape)
    
    save_path = src.split(".")[0]+"randdim1Aug.mat"
    print('savepath:',save_path)
    with h5py.File(save_path,"w") as f:
        f.create_dataset('V_pinv_LS_all_data',data=data_saved_V)
        f.create_dataset('H_est_LS_all_data',data=data_saved_x)
        f.create_dataset("H_all_data",data=data_saved_y)
    
    print("Data saved successfully to", save_path)    
    










if __name__ =="__main__":
    path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/fixed_UEs/train_Dataset_dB-15_N36_K4_L8_S9_Setup200_Reliz1000.mat'
    data_preAugment(path)