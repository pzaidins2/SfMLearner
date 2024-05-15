from __future__ import division
import tensorflow as tf
import numpy as np
import os
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

flags = tf.compat.v1.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def main(_):
    tf.compat.v1.disable_eager_execution()
    with open('SfMLearner/data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=3*FLAGS.batch_size,
                        mode='depth')
    saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    val_file = os.path.join( FLAGS.dataset_dir, "val.txt" )
    dir_lst = [ ]
    with open( val_file, 'r' ) as f:
        for line in f.readlines():
            split_line = line.split()
            dir_lst.append( split_line[ 0 ] + "/" + split_line[ 1 ] + ".jpg" )

    path_lst = [ *map( lambda x: os.path.join( FLAGS.dataset_dir, x ), dir_lst ) ]
    N = len( path_lst )

    with tf.compat.v1.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        for t in range(0, N, 3*FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, N))
            inputs = np.zeros(
                (3*FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(3*FLAGS.batch_size):
                b_i = b % 3
                b_j = b // 3
                idx = t + b_j
                if idx >= N:
                    break
                # fh = open( path_lst[idx], 'r')

                raw_im = pil.open(path_lst[idx])
                scaled_im = raw_im.resize((3*FLAGS.img_width, FLAGS.img_height), pil.BILINEAR)

                # print(np.array(scaled_im).shape)
                inputs[b] = np.array(scaled_im)[:,FLAGS.img_width*b_i:FLAGS.img_width*(b_i+1)]
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sfm.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= N:
                    break
                pred_all.append(pred['depth'][b,:,:,0])
        np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.compat.v1.app.run()
