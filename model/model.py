import tensorflow as tf
import numpy as np
import ops
import time
import os
from dataloader import Dataloader


class CCNN(object):

    def __init__(self, sess, epoch=14, initial_learning_rate=0.003, batch_size=1, patch_size=1, isTraining=False, model_name='CCNN.model'):
        self.sess = sess
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.initial_learning_rate = initial_learning_rate
        self.isTraining = isTraining
        self.model_name = model_name
        self.epoch=epoch
        self.model_collection = ['CCNN']

        self.build_CCNN()

        if self.isTraining == 'True':
            self.build_losses()
            self.build_summaries()

    def build_CCNN(self):
        print(" [*] Building CCNN model...")

        if(self.isTraining=='True'):
            self.disp = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='disparity') / 255.0
            self.gt = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='gt')
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
        else:
            self.disp = tf.placeholder(tf.float32, name='disparity') / 255.0

        with tf.variable_scope('CCNN'):

            with tf.variable_scope("conv1"):
                self.conv1 = ops.conv2d(self.disp, [3, 3, 1, 64], 1, True, padding='VALID')

            with tf.variable_scope("conv2"):
                self.conv2 = ops.conv2d(self.conv1, [3, 3, 64, 64], 1, True, padding='VALID')

            with tf.variable_scope("conv3"):
                self.conv3 = ops.conv2d(self.conv2, [3, 3, 64, 64], 1, True, padding='VALID')

            with tf.variable_scope("conv4"):
                self.conv4 = ops.conv2d(self.conv3, [3, 3, 64, 64], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_1"):
                self.fc1 = ops.conv2d(self.conv4, [1, 1, 64, 100], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_2"):
                self.fc2 = ops.conv2d(self.fc1, [1, 1, 100, 100], 1, True, padding='VALID')

            with tf.variable_scope("prediction"):
                self.prediction = ops.conv2d(self.fc2, [1, 1, 100, 1], 1, False, padding='VALID')

            with tf.variable_scope("variables"):
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

        print(' [*] Loading training set...')
        dataloader = Dataloader(file=args.dataset_training)
        disp_files, gt_files = dataloader.read_list_file()

        print(' [*] Training data loaded successfully')
        epoch = 0
        iteration = 0
        lr = self.initial_learning_rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        self.sess.run(init_op)

        print(" [*] Start Training...")
        while epoch < self.epoch:
            for i, item in enumerate(disp_files):
                print(" [*] Loading train image: " + disp_files[i])
                disp_patches, gt_patches = dataloader.get_training_patches(disp_files[i], gt_files[i], self.patch_size)
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

            if epoch == 10:
                lr = lr/10

        coord.request_stop()
        coord.join(threads)

    def test(self, args):
        print("[*] Testing....")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        self.saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        if args.checkpoint_path != '':
            self.saver.restore(self.sess, args.checkpoint_path)
            print(" [*] Load model: SUCCESS")
        else:
            print(" [*] Load failed...neglected")
            print(" [*] End Testing...")
            raise ValueError('self.checkpoint_path == ')

        dataloader = Dataloader(file=args.dataset_testing)
        disp_files, gt_files = dataloader.read_list_file()

        self.prediction = tf.pad(tf.nn.sigmoid(self.prediction), tf.constant([[0, 0], [4, 4,], [4, 4], [0, 0]]), "CONSTANT")
        self.confmap_png = tf.image.encode_png(tf.cast(tf.scalar_mul(65535.0, tf.squeeze(self.prediction, axis=0)), dtype=tf.uint16))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        self.sess.run(init_op)

        print(" [*] Start Testing...")
        for i, item in enumerate(disp_files):
            print(" [*] Loading test image:" + disp_files[i])
            disp_patches = dataloader.get_testing_image(disp_files[i])
            batch_disp = self.sess.run([disp_patches])
            start = time.time()
            prediction = self.sess.run(self.confmap_png,feed_dict={self.disp: batch_disp})
            current = time.time()
            output_file = args.output_path + disp_files[i].strip().split('/')[-1]
            self.sess.run(tf.write_file(output_file, prediction))
            print(" [*] CCNN confidence prediction saved in:" + output_file)
            print(" [*] CCNN running time:" + str(current - start) + "s")

        coord.request_stop()
        coord.join(threads)


    def build_summaries(self):
        tf.summary.scalar('loss', self.loss, collections=self.model_collection)
        tf.summary.scalar('learning_rate', self.learning_rate, collections=self.model_collection)




