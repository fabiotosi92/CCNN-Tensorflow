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
        self.radius = int(patch_size/2)
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

        kernel_size = 3
        filters = 64
        fc_filters = 100

        with tf.variable_scope('CCNN'):

            with tf.variable_scope("conv1"):
                self.conv1 = ops.conv2d(self.disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv2"):
                self.conv2 = ops.conv2d(self.conv1, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv3"):
                self.conv3 = ops.conv2d(self.conv2, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv4"):
                self.conv4 = ops.conv2d(self.conv3, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_1"):
                self.fc1 = ops.conv2d(self.conv4, [1, 1, filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_2"):
                self.fc2 = ops.conv2d(self.fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("prediction"):
                self.prediction = ops.conv2d(self.fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

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

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print(' [*] Loading training set...')
        dataloader = Dataloader(file=args.dataset_training, isTraining=self.isTraining)
        patch_disp, patch_gt = dataloader.get_training_patches(self.patch_size, args.threshold)
        line = dataloader.disp_filename

        print(' [*] Training data loaded successfully')
        epoch = 0
        iteration = 0
        lr = self.initial_learning_rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_samples = dataloader.count_text_lines(args.dataset_training)

        print(" [*] Start Training...")
        while epoch < self.epoch:
            for i in range(num_samples):
                batch_disp, batch_gt, filename = self.sess.run([patch_disp, patch_gt, line])
                print(" [*] Training image: " + filename)

                step_image = 0
                while step_image < len(batch_disp):
                    offset = (step_image * self.batch_size) % (batch_disp.shape[0] - self.batch_size)
                    batch_data = batch_disp[offset:(offset + self.batch_size), :, :, :]
                    batch_labels = batch_gt[offset:(offset +  self.batch_size), self.radius:self.radius+1, self.radius:self.radius+1, :]

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

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if args.checkpoint_path != '':
            self.saver.restore(self.sess, args.checkpoint_path)
            print(" [*] Load model: SUCCESS")
        else:
            print(" [*] Load failed...neglected")
            print(" [*] End Testing...")
            raise ValueError('self.checkpoint_path == ')

        dataloader = Dataloader(file=args.dataset_testing, isTraining=self.isTraining)
        disp_batch = dataloader.disp
        line = dataloader.disp_filename
        prediction = tf.pad(tf.nn.sigmoid(self.prediction), tf.constant([[0, 0], [self.radius, self.radius,], [self.radius, self.radius], [0, 0]]), "CONSTANT")
        png = tf.image.encode_png(tf.cast(tf.scalar_mul(65535.0, tf.squeeze(prediction, axis=0)), dtype=tf.uint16))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_samples = dataloader.count_text_lines(args.dataset_testing)

        print(" [*] Start Testing...")
        for step in range(num_samples):
            batch, filename = self.sess.run([disp_batch, line])
            print(" [*] Test image:" + filename)
            start = time.time()
            confidence = self.sess.run(png,feed_dict={self.disp: batch})
            current = time.time()
            output_file = args.output_path + filename.strip().split('/')[-1]
            self.sess.run(tf.write_file(output_file, confidence))
            print(" [*] CCNN confidence prediction saved in:" + output_file)
            print(" [*] CCNN running time:" + str(current - start) + "s")

        coord.request_stop()
        coord.join(threads)


    def build_summaries(self):
        tf.summary.scalar('loss', self.loss, collections=self.model_collection)
        tf.summary.scalar('learning_rate', self.learning_rate, collections=self.model_collection)




