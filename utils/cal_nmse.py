from tools import *

#----------------------------------------------------compare SIM layer---------------------------------------------
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/v1_test_Dataset_dB-25_N36_K4_L2_S9_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/valid_Dataset_dB-25_N36_K4_L3_S9_Setup10_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L5_S9_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000.mat'

#----------------------------------------------------compare Subframe layer---------------------------------------------
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.mat'
valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/v1_test_Dataset_dB-25_N36_K4_L4_S10_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.mat'
# valid_path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset10_N36K4_3types/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.mat'

nmse = cal_NMSE_by_matpath_h5(valid_path,"H_est_MMSE_all_data")
print("NMMSE of valid dataset::",nmse)