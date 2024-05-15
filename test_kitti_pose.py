from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
import imageio

flags = tf.compat.v1.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        else:
            image_seq = np.hstack((image_seq, curr_img))
    return image_seq

def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    # TODO: unnecessary to check if the drives match
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

def main():
    tf.compat.v1.disable_eager_execution()
    sfm = SfMLearner()
    sfm.setup_inference( FLAGS.img_height,
                         FLAGS.img_width,
                         'pose',
                         FLAGS.seq_length )
    for var in tf.compat.v1.trainable_variables():
        print( var )
    saver = tf.compat.v1.train.Saver( [ var for var in tf.compat.v1.trainable_variables() ] )

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    # seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
    # img_dir = os.path.join(seq_dir, 'image_2')
    val_file = os.path.join(FLAGS.dataset_dir,"val.txt")
    dir_lst = []
    with open(val_file, 'r') as f:
        for line in f.readlines():
            split_line = line.split()
            dir_lst.append( split_line[0]+"/"+split_line[1]+".jpg")

    path_lst =  [*map(lambda x: os.path.join(FLAGS.dataset_dir,x), dir_lst)]
    N = len(path_lst)

    max_src_offset = (FLAGS.seq_length - 1)//2
    with tf.compat.v1.Session() as sess:

        saver.restore(sess, FLAGS.ckpt_file)
        for tgt_idx in range(N):
            if tgt_idx % 100 == 0:
                print('Progress: %d/%d' % (tgt_idx, N))
            # TODO: currently assuming batch_size = 1
            image_file = path_lst[tgt_idx]
            image_seq = imageio.v2.imread(image_file)
            pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
            pred_poses = pred['pose'][0]
            # Insert the target pose [0, 0, 0, 0, 0, 0] 
            pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)
            out_file = FLAGS.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_poses,[*range(FLAGS.seq_length)])

main()
