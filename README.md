# CCNN-Tensorflow

Tensorflow implementation of confidence estimation using a convolutional neural network

**Learning from scratch a confidence measure**  
[Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/) and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
BMVC 2016

For more details:  
[project page](https://vision.disi.unibo.it/~mpoggi/code.html)  
[pdf](https://vision.disi.unibo.it/~mpoggi/papers/bmvc2016.pdf)  

## Requirements
This code was tested with Tensorflow 1.4, CUDA 8.0 and Ubuntu 16.04.  

## Training

Training takes about 15 minutes with the default parameters on 20 images of **KITTI 2012** on a single 1080Ti GPU card. 

```shell
python ./model/main.py --isTraining True --epoch 14 --batch_size 64 --patch_size 9 --dataset_training ./utils/kitti_training_set.txt --initial_learning_rate 0.003 --log_directory ./log --save_epoch_freq 2 --model_name CCNN.model 
```

**Warning:** appropriately change of ./utils/kitti_training_set.txt is necessary to train from scratch the network. To this aim, it's provided a shell script to generate a new list file. 

```shell
./utils/kitti_generate_file.sh [path_disparities] [path_kitti_groundtruth] [index_from] [index_to] [output_file]
```

## Testing 

Test takes about 0.03 seconds on a single image of **KITTI 2012**  using a 1080Ti GPU card. 

```shell
python ./model/main.py --isTraining False --batch_size 1 --dataset_testing ./utils/kitti_testing_set.txt --checkpoint_path ./log/CCNN.model-595140 --image_width 1320 --image_height 390 --output_path ./output/CCNN/ad-census/

## Models

You can download a pre-trained model in ./log

The model was trained for 14 epochs, a batch size of 64, an initial learning rate of 0.003 (reduced to 0.0003 after 10 epochs) and patches of 9x9.

```
