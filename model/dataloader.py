import tensorflow as tf


class Dataloader(object):

    def __init__(self, file, isTraining):
        self.file = file
        self.isTraining = isTraining

        self.disp  = None
        self.gt = None

        input_queue = tf.train.string_input_producer([self.file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line], ';').values

        if isTraining == 'True':
            self.disp = tf.cast(self.read_image(split_line[0], [None, None, 1]), tf.float32)
            self.gt = tf.cast(self.read_image(split_line[1], [None, None, 1], dtype=tf.uint16), tf.float32)/ 256.0
            self.disp_filename = split_line[0]
            self.gt_filename = split_line[1]
        else:
            self.disp = tf.stack([tf.cast(self.read_image(split_line[0], [None, None, 1]), tf.float32)], 0)
            self.disp_filename = split_line[0]

    def get_training_patches(self, patch_size, threshold):
        disp_list = []
        gt_list = []

        disp_list.append(self.disp)
        gt_list.append(self.gt)

        disp_patches = tf.reshape(
            tf.extract_image_patches(images=disp_list, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                     padding='VALID'), [-1, patch_size, patch_size, 1])

        gt_patches = tf.reshape(
            tf.extract_image_patches(images=gt_list, ksizes=[1, patch_size, patch_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                     padding='VALID'), [-1, patch_size, patch_size, 1])

        mask = gt_patches[:, int(patch_size/2):int(patch_size/2)+1, int(patch_size/2):int(patch_size/2)+1, :] > 0
        valid = tf.tile(mask, [1, patch_size, patch_size, 1])

        disp_patches = tf.reshape(tf.boolean_mask(disp_patches, valid), [-1, patch_size, patch_size, 1])
        gt_patches = tf.reshape(tf.boolean_mask(gt_patches, valid), [-1, patch_size, patch_size, 1])

        labels = tf.cast(tf.abs(disp_patches - gt_patches) <= threshold, tf.float32)

        return disp_patches, labels

    def read_image(self, image_path, shape=None, dtype=tf.uint8):
        image_raw = tf.read_file(image_path)
        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)
        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)
        return image

    def count_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)






