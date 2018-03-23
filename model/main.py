import argparse
import tensorflow as tf
from model import CCNN

parser = argparse.ArgumentParser(description='Argument parser')

"""Arguments related to run mode"""
parser.add_argument('--isTraining', dest='isTraining', type=str, default='False', help='train, test')

"""Arguments related to training"""
parser.add_argument('--epoch',  dest='epoch', type=int, default=14, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=9, help='# images in patches')
parser.add_argument('--dataset_training',  dest='dataset_training', type=str, default='../utils/kitti_training_set.txt', help='dataset training')
parser.add_argument('--dataset_testing', dest='dataset_testing', type=str, default='../utils/kitti_testing_set.txt', help='dataset testing')
parser.add_argument('--initial_learning_rate', dest='initial_learning_rate', type=float, default=0.003, help='initial learning rate for gradient descent')
parser.add_argument('--threshold', dest='threshold', type=float, default=3, help='disparity error if absolute difference between disparity and groundtruth > threshold')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--log_directory', dest='log_directory', type=str, default='../log', help='directory to save checkpoints and summaries')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default='', help='path to a specific checkpoint to load' )
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=1, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--model_name', dest='model_name', type=str, default='CCNN.model', help='model name')
parser.add_argument('--output_path', dest='output_path', type=str, default='../output/CCNN/', help='model name')

args = parser.parse_args()


def main(_):
    with tf.Session() as sess:
        model = CCNN(sess,
                     isTraining=args.isTraining,
                     epoch=args.epoch,
                     batch_size=args.batch_size,
                     patch_size=args.patch_size,
                     initial_learning_rate=args.initial_learning_rate,
                     model_name=args.model_name
                     )

        if args.isTraining == 'True':
            model.train(args)
        else:
            model.test(args)


if __name__ == '__main__':
    tf.app.run()
