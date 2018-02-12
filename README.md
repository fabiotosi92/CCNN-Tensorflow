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

```shell
python ./model/main.py --isTraining True --epoch 14 --batch_size 64 --patch_size 9 --dataset_training ./utils/kitti_training_set.txt --initial_learning_rate 0.003 --log_directory ./log --save_epoch_freq 2 --model_name CCNN.model 
```

## Testing 

```shell
python ./model/main.py --isTraining False --batch_size 1 --dataset_testing ./utils/kitti_testing_set.txt --checkpoint_path ./log/CCNN.model-595140 --image_width 1320 --image_height 390 --output_path ./output/CCNN/
```
