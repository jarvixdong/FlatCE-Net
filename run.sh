# LOG_PATH="logger/240225/CDRN_B3D18C64_test_Dataset_dB-15_N36_K4_L5_S9_Setup10_Reliz1000.log"
LOG_PATH="logger/240225/flatCE_L3C16_test_Dataset_dB-15_N36_K4_L5_S9_Setup20_Reliz1000_v4.log"
# LOG_PATH="logger/230225/LeNet_L3C512_test_Dataset_dB-10_N36_K4_L2_S9_Setup10_Reliz1000.log"
config='conf/config_multisetup.yml'
nohup python train.py --log_path "$LOG_PATH" --config "$config"> "$LOG_PATH" 2>&1 &