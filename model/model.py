import tensorflow as tf
import numpy as np
import ops
import time
import os
from dataloader import Dataloader

class CCNN(object):

    def __init__(self, sess, image_width=9, image_height=9, epoch=14, initial_learning_rate=0.003, batch_size=1, patch_size=1, isTraining=False, model_name='CCNN.model'):
        self.sess = sess
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.initial_learning_rate = initial_learning_rate
        self.isTraining = isTraining
        self.model_name = model_name
        self.epoch=epoch
        self.model_collection = ['CCNN']

        self.build_CCNN()
        if (self.isTraining == 'True'):
            self.build_losses()
            self.build_summaries()


    def build_CCNN(self):
        print(" [*] Building CCNN model...\n")

        if(self.isTraining=='True'):
            self.disp = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='disparity') / 255.0
            self.gt = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='gt')
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
        else:
            self.disp = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, 1], name='disparity') / 255.0

        with tf.variable_scope('CCNN'):
            print ('input shape:')
            print(self.disp.get_shape().as_list())

            with tf.variable_scope("conv1"):
                self.conv1 = ops.conv2d(self.disp, [3, 3, 1, 64], 1, True, padding='VALID')
                print ('conv1:')
                print(self.conv1.get_shape().as_list())

            with tf.variable_scope("conv2"):
                self.conv2 = ops.conv2d(self.conv1, [3, 3, 64, 64], 1, True, padding='VALID')
                print ('conv2:')
                print(self.conv2.get_shape().as_list())

            with tf.variable_scope("conv3"):
                self.conv3 = ops.conv2d(self.conv2, [3, 3, 64, 64], 1, True, padding='VALID')
                print ('conv3:')
                print(self.conv3.get_shape().as_list())

            with tf.variable_scope("conv4"):
                self.conv4 = ops.conv2d(self.conv3, [3, 3, 64, 64], 1, True, padding='VALID')
                print ('conv4:')
                print(self.conv4.get_shape().as_list())

            with tf.variable_scope("fully_connected_1"):
                self.fc1 = ops.conv2d(self.conv4, [1, 1, 64, 100], 1, True, padding='VALID')
                print ('fc1:')
                print(self.fc1.get_shape().as_list())

            with tf.variable_scope("fully_connected_2"):
                self.fc2 = ops.conv2d(self.fc1, [1, 1, 100, 100], 1, True, padding='VALID')
                print ('fc2:')
                print(self.fc2.get_shape().as_list())

            with tf.variable_scope("prediction"):
                self.prediction = ops.conv2d(self.fc2, [1, 1, 100, 1], 1, False, padding='VALID')
                print ('prediction:')
                print(self.prediction.get_shape().as_list())

            with tf.variable_scope("variables") as scope:
                t_vars = tf.trainable_variables()
                self.vars = [var for var in t_vars]


    def build_losses(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt, logits=self.prediction))


    def train(self, args):
        print("\n [*] Training....")

        if not os.path.exists(args.log_directory):
            os.makedirs(args.log_directory)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=self.vars)
        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all(self.model_collection[0])
        self.writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=self.sess.graph)

        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()

        print(" [*] Number of trainable parameters: {}".format(total_num_parameters))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

        print(' [*] Loading training set...')
        dataloader = Dataloader(file=args.dataset_training)
        left_files, gt_files = dataloader.read_list_file()

        print(' [*] Training data loaded successfully')
        epoch = 0
        iteration = 0
        lr = self.initial_learning_rate

        print(" [*] Start Training...")
        while epoch < self.epoch:
            for i, item in enumerate(left_files):
                print(" [*] Loading train image: " + left_files[i])
                disp_patches, gt_patches = dataloader.get_training_patches(left_files[i], gt_files[i], self.patch_size)
                batch_disp, batch_gt = self.sess.run([disp_patches, gt_patches])

                step_image = 0
                while step_image < len(batch_disp):
                    offset = (step_image * self.batch_size) % (batch_disp.shape[0] - self.batch_size)
                    batch_data = batch_disp[offset:(offset + self.batch_size), :, :, :]
                    batch_labels = batch_gt[offset:(offset +  self.batch_size), int(self.patch_size/2):int(self.patch_size/2)+1, int(self.patch_size/2):int(self.patch_size/2)+1, :]

                    _, loss, summary_str = self.sess.run([self.optimizer, self.loss, self.summary_op], feed_dict={self.disp:batch_data, self.gt:batch_labels, self.learning_rate: lr})

                    print("Epoch: [%2d]" % epoch + ", Image: [%2d]" % i + ", Iter: [%2d]" % iteration + ", Loss: [%2f]" % loss )
                    self.writer.add_summary(summary_str, global_step=iteration)
                    iteration = iteration + 1
                    step_image = step_image + self.batch_size

            epoch = epoch + 1

            if np.mod(epoch, args.save_epoch_freq) == 0:
                self.saver.save(self.sess, args.log_directory + '/' + self.model_name, global_step=iteration)
            if(epoch==10):
                lr = lr/10


    def test(self, args):
        print("\n [*] Testing....")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        W = tf.placeholder(tf.int32)
        H = tf.placeholder(tf.int32)

        self.saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

        if args.checkpoint_path != '':
            self.saver.restore(self.sess, args.checkpoint_path)
            print(" [*] Load model: SUCCESS")
        else:
            print(" [*] Load failed...neglected")
            print(" [*] End Testing...")
            raise ValueError('self.checkpoint_path == ')

        dataloader = Dataloader(file=args.dataset_testing)
        left_files, gt_files = dataloader.read_list_file()
        self.prediction_resized = tf.nn.sigmoid(tf.image.resize_image_with_crop_or_pad(image=self.prediction, target_height=H, target_width=W))

        print(" [*] Start Testing...")
        for i, item in enumerate(left_files):
            print(" [*] Loading test image:" + left_files[i])
            disp_patches = dataloader.get_testing_image(left_files[i])
            batch_disp = self.sess.run([disp_patches])
            shape = batch_disp[0].shape
            disp_patches = tf.image.resize_image_with_crop_or_pad(image=disp_patches, target_height=self.image_height, target_width=self.image_width)
            batch_disp = self.sess.run([disp_patches])
            start = time.time()
            prediction = self.sess.run(self.prediction_resized,feed_dict={self.disp: batch_disp, H:shape[0], W:shape[1]})
            current = time.time()
            confmap_png = tf.image.encode_png(tf.cast(tf.scalar_mul(65535.0, tf.squeeze(prediction, axis=0)), dtype=tf.uint16))
            output_file = args.output_path + left_files[i].strip().split('/')[-1]
            self.sess.run(tf.write_file(output_file, confmap_png))
            print(" [*] CCNN confidence prediction saved in:" + output_file)
            print(" [*] CCNN running time:" + str(current - start) + "s")

    def build_summaries(self):
        tf.summary.scalar('loss', self.loss, collections=self.model_collection)
        tf.summary.scalar('learning_rate', self.learning_rate, collections=self.model_collection)




