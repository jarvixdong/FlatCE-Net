# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L3_S9_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-25_N36_K4_L4_S9_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/test_Dataset_dB-25_N36_K4_L5_S9_Setup50_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-25_N36_K4_L5_S9_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_lr1e-3.log 2>&1 & echo $! >> logger/180225//v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_lr1e-3.log
# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_CDRN_ep100.log 2>&1 & echo $! >> logger/180225//v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_CDRN_ep100.log

# nohup python train.py > logger/180225/v1_valid_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000.log 2>&1 & echo $! >> logger/180225/v1_valid_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000.log
# nohup python train.py > logger/180225/v1_valid_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_CDRN.log 2>&1 & echo $! >> logger/180225/v1_valid_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_CDRN.log

# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_C64L3.log 2>&1 & echo $! >> logger/180225/v1_test_Dataset_dB-25_N36_K4_L6_S9_Setup50_Reliz1000_C64L3.log


# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S10_Setup50_Reliz1000.log 2>&1 & echo $! >> logger/180225//train.log
# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S10_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S11_Setup50_Reliz1000.log 2>&1 & echo $! >> logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S11_Setup50_Reliz1000.log
# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S11_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S12_Setup50_Reliz1000.log 2>&1 & echo $! >> logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S12_Setup50_Reliz1000.log
# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S12_Setup50_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S13_Setup50_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/v1_test_Dataset_dB-25_N36_K4_L4_S13_Setup50_Reliz1000_CDRN.log 2>&1 &


# nohup python train.py > logger/180225/test_Dataset_dB-20_N36_K4_L4_S9_Setup10_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-20_N36_K4_L4_S9_Setup10_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/test_Dataset_dB-15_N36_K4_L4_S9_Setup10_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-15_N36_K4_L4_S9_Setup10_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/test_Dataset_dB-10_N36_K4_L4_S9_Setup10_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-10_N36_K4_L4_S9_Setup10_Reliz1000_CDRN.log 2>&1 &

# nohup python train.py > logger/180225/test_Dataset_dB-5_N36_K4_L4_S9_Setup10_Reliz1000.log 2>&1 &
# nohup python train.py > logger/180225/test_Dataset_dB-5_N36_K4_L4_S9_Setup10_Reliz1000_CDRN.log 2>&1 &



#--------------------------------------------------------------------------------------- 20dBm-----------------------------------------------------------------
# LOG_PATH="logger/210225/v1_test_Dataset_dB-10_N36_K4_L2_S9_Setup50_Reliz1000_lrS_1e-2_L4C32_p20_adam.log"
# LOG_PATH="logger/220225/CDRN_b3d18c64_Step50lr1e-2_test_Dataset_dB-10_N36_K4_L2_S9_Setup10_Reliz1000.log"
LOG_PATH="test_train.log"
nohup python train.py > "$LOG_PATH" 2>&1 &
# echo $! >> "$LOG_PATH"
