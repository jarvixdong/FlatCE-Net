#!/bin/bash

CONFIG_PATH="conf/config_multisetup.yml"
# SAVE_PATH="/mnt/parscratch/users/elq20xd/channel_estimation/save_models/fixed_UEs_15GHz/flatCE_L4C16_valid_Dataset_dB-15_N36_K4_L4_S9_Setup20_Reliz1000Ran1/"
# SAVE_PATH="/mnt/parscratch/users/elq20xd/channel_estimation/save_models/fixed_UEs_15GHz/CDRN_B3D18C64_valid_Dataset_dB-15_N36_K4_L8_S9_Setup20_Reliz1000Ran1/"
SAVE_PATH="/mnt/parscratch/users/elq20xd/channel_estimation/save_models/fixed_UEs_15GHz_ReduceLR/valid_Dataset_dB-15_N36_K4_L4_S10_Setup20_Reliz1000Ran1_"
# SAVE_PATH="${SAVE_PATH}flatCE_l2c32dia6_sp"   # DnCNN_MultiBlock_ds flatCE_l3c16dia6
SAVE_PATH="${SAVE_PATH}attn_FeatureCNN"

LOG_PATH="${SAVE_PATH}/train.log"

# Create directories if they don’t exist
if [ -d "${SAVE_PATH}" ]; then
    echo "SAVE_PATH already exists"
    echo -n "Continue or quit? [y/n]: "
    read -r cmd
    if [ "$cmd" == "n" ]; then
        exit 0
    fi
else
    mkdir -p "${SAVE_PATH}/save_models/"
    
fi

# Copy config file to save_path
cp "${CONFIG_PATH}" "${SAVE_PATH}/"

# Run the training script with nohup, passing save_path as an argument
nohup python train.py --log_path "${LOG_PATH}" --config "${CONFIG_PATH}" --save_path "${SAVE_PATH}" > "${LOG_PATH}" 2>&1 &
# python train.py --log_path "${LOG_PATH}" --config "${CONFIG_PATH}" --save_path "${SAVE_PATH}"