from __future__ import division
import tensorflow as tf
import tf_slim as slim
import numpy as np

# CODED MODIFIED FROM nets.py FOR "ORIGINAL IMPLEMENTATION"
# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.compat.v1.image.resize_nearest_neighbor(inputs, [rH, rW])

def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1]#.value
    W = inputs.get_shape()[2]#.value
    num_source = int(src_image_stack.get_shape()[3]//3)
    with tf.compat.v1.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.compat.v1.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                        normalizer_fn=None, activation_fn=None)
                    
                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points
def residual_block(input, num_outputs, kernel_size, stride, scope):
    with tf.compat.v1.variable_scope(scope):
        net = slim.conv2d(input, num_outputs, kernel_size, stride=stride, scope='conv1')
        net = slim.conv2d(net, num_outputs, kernel_size, stride=1, activation_fn=None, scope='conv2')
        if input.get_shape()[-1] != num_outputs or stride != 1:
            input = slim.conv2d(input, num_outputs, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        net = tf.nn.relu(net + input)
    return net

# ALL FOR DENSE NET BELOW
def dense_block(input, num_layers, scope,is_training=True):
    k=32 #hardcoded
    with tf.compat.v1.variable_scope(scope):
        layers_concat = [input]
        x = input
        for i in range(num_layers):
            x = batch_norm(x, scope='bn_%d' % i,is_training=is_training)
            x = slim.conv2d(x, k, [3, 3], scope='conv_%d' % i)
            layers_concat.append(x)
            x = tf.concat(layers_concat, axis=-1)
    return x
def batch_norm(input, scope, is_training=True, reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x = tf.compat.v1.layers.batch_normalization(
            inputs=input,
            momentum=0.9,
            epsilon=1e-5,
            center=True,
            scale=True,
            training=is_training,
            fused=True
        )
        return tf.nn.relu(x)
def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH != rH or iW != rW:
        inputs = tf.image.resize(inputs, [rH, rW])
    return inputs

def transition_layer(input, num_outputs, scope, is_training=True, reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x = batch_norm(input, scope='bn_%d' % num_outputs, is_training=is_training)
        x = tf.compat.v1.layers.conv2d(x, num_outputs, [1, 1], name='conv')
        x = tf.compat.v1.layers.average_pooling2d(x, [2, 2], strides=2, name='pool')
    return x



def disp_net(tgt_image, struct, is_training=True):
    H = tgt_image.get_shape()[1]#.value
    W = tgt_image.get_shape()[2]#.value
    if struct ==0:
        with tf.compat.v1.variable_scope('depth_net',reuse=tf.compat.v1.AUTO_REUSE) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
                cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
                cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
                cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
                cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
                cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
                cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
                cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
                cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

                upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
                # There might be dimension mismatch due to uneven down/up-sampling
                upcnv7 = resize_like(upcnv7, cnv6b)
                i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
                icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

                upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
                upcnv6 = resize_like(upcnv6, cnv5b)
                i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
                icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

                upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv5 = resize_like(upcnv5, cnv4b)
                i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
                icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
                i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
                icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
                disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
                disp4_up = tf.compat.v1.image.resize_bilinear(disp4, [np.int64(H/4), np.int64(W/4)])

                upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
                icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
                disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
                disp3_up = tf.compat.v1.image.resize_bilinear(disp3, [np.int64(H/2), np.int64(W/2)])

                upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
                i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
                icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
                disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
                disp2_up = tf.compat.v1.image.resize_bilinear(disp2, [H, W])

                upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
                i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
                icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
                disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
                
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    elif struct ==1:
        with tf.compat.v1.variable_scope('depth_net') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                cnv1 = residual_block(tgt_image, 32, [7, 7], stride=2, scope='cnv1')
                cnv2 = residual_block(cnv1, 64, [5, 5], stride=2, scope='cnv2')
                cnv3 = residual_block(cnv2, 128, [3, 3], stride=2, scope='cnv3')
                cnv4 = residual_block(cnv3, 256, [3, 3], stride=2, scope='cnv4')
                cnv5 = residual_block(cnv4, 512, [3, 3], stride=2, scope='cnv5')
                cnv6 = residual_block(cnv5, 512, [3, 3], stride=2, scope='cnv6')
                cnv7 = residual_block(cnv6, 512, [3, 3], stride=2, scope='cnv7')

                upcnv7 = slim.conv2d_transpose(cnv7, 512, [3, 3], stride=2, scope='upcnv7')
                upcnv7 = resize_like(upcnv7, cnv6)
                i7_in = tf.concat([upcnv7, cnv6], axis=3)
                icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

                upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
                upcnv6 = resize_like(upcnv6, cnv5)
                i6_in = tf.concat([upcnv6, cnv5], axis=3)
                icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

                upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv5 = resize_like(upcnv5, cnv4)
                i5_in = tf.concat([upcnv5, cnv4], axis=3)
                icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
                i4_in = tf.concat([upcnv4, cnv3], axis=3)
                icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
                disp4 = DISP_SCALING * slim.conv2d(icnv4, 1, [3, 3], stride=1,
                                                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
                disp4_up = tf.compat.v1.image.resize_bilinear(disp4, [H // 4, W // 4])

                upcnv3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
                i3_in = tf.concat([upcnv3, cnv2, disp4_up], axis=3)
                icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
                disp3 = DISP_SCALING * slim.conv2d(icnv3, 1, [3, 3], stride=1,
                                                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
                disp3_up = tf.compat.v1.image.resize_bilinear(disp3, [H // 2, W // 2])

                upcnv2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
                i2_in = tf.concat([upcnv2, cnv1, disp3_up], axis=3)
                icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
                disp2 = DISP_SCALING * slim.conv2d(icnv2, 1, [3, 3], stride=1,
                                                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
                disp2_up = tf.compat.v1.image.resize_bilinear(disp2, [H, W])

                upcnv1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
                i1_in = tf.concat([upcnv1, disp2_up], axis=3)
                icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
                disp1 = DISP_SCALING * slim.conv2d(icnv1, 1, [3, 3], stride=1,
                                                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    else:
        is_training=True
        with tf.compat.v1.variable_scope('depth_net', reuse=tf.compat.v1.AUTO_REUSE) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.relu,
                                outputs_collections=end_points_collection):
                cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
                cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
                cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
                cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
                cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
                cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
                cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
                cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
                cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
                cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
                cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
                cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
                cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
                cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

                upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
                # There might be dimension mismatch due to uneven down/up-sampling
                upcnv7 = resize_like(upcnv7, cnv6b)
                i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
                icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

                upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
                upcnv6 = resize_like(upcnv6, cnv5b)
                i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
                icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

                upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
                upcnv5 = resize_like(upcnv5, cnv4b)
                i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
                icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

                upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
                i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
                icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
                disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
                disp4_up = tf.compat.v1.image.resize_bilinear(disp4, [np.int64(H/4), np.int64(W/4)])

                upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
                icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
                disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
                disp3_up = tf.compat.v1.image.resize_bilinear(disp3, [np.int64(H/2), np.int64(W/2)])

                upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
                i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
                icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
                disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
                disp2_up = tf.compat.v1.image.resize_bilinear(disp2, [np.int64(H/1), np.int64(W/1)])

                upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
                i1_in  = tf.concat([upcnv1, tgt_image, disp2_up], axis=3)
                icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
                disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
                
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return [disp1, disp2, disp3, disp4], end_points
        

