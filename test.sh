#!bin/bash

python ./model/main.py --isTraining False --batch_size 1 --dataset_testing ./utils/kitti_testing_set.txt --checkpoint_path ./log/CCNN.model-595140 --image_width 1320 --image_height 390 --output_path ./output/CCNN/
