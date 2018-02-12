#!bin/bash

python ./model/main.py --isTraining True --epoch 14 --batch_size 64 --patch_size 9 --dataset_training ./utils/kitti_training_set.txt --initial_learning_rate 0.003 --log_directory ./log --save_epoch_freq 2 --model_name CCNN.model 
