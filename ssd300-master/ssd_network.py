import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.keras.initializers import he_normal
from set_flags_variables import *

def fully_connected(prev_layer, num_units, is_training=False):
    layer = tf.layers.dense(prev_layer, num_units, use_bias=True, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer

#one 3*3 conv layer
def conv_layer_conv(prev_layer, layer_depth, is_training=False):
    conv_layer1 = tf.layers.conv2d(prev_layer, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer1_bn = tf.layers.batch_normalization(conv_layer1, training=is_training)
    conv_layer1_out = tf.nn.relu(conv_layer1_bn)

    pool_layer1 = tf.layers.max_pooling2d(conv_layer1_out,[2,2],strides=2,padding='same')
    return pool_layer1

#double 3*3 conv layers
def conv_layer_2conv(prev_layer, layer_depth, is_training=False):
    conv_layer1 = tf.layers.conv2d(prev_layer, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer1_bn = tf.layers.batch_normalization(conv_layer1, training=is_training)
    conv_layer1_out = tf.nn.relu(conv_layer1_bn)
    
    conv_layer2 = tf.layers.conv2d(conv_layer1_out, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer2_bn = tf.layers.batch_normalization(conv_layer2, training=is_training)
    conv_layer2_out = tf.nn.relu(conv_layer2_bn)

    pool_layer2 = tf.layers.max_pooling2d(conv_layer2_out,[2,2],strides=2,padding='same')
    return pool_layer2,conv_layer2_out

#triple 3*3 conv layers
def conv_layer_3conv(prev_layer, layer_depth, is_training=False):
    conv_layer1 = tf.layers.conv2d(prev_layer, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer1_bn = tf.layers.batch_normalization(conv_layer1, training=is_training)
    conv_layer1_out = tf.nn.relu(conv_layer1_bn)
    
    conv_layer2 = tf.layers.conv2d(conv_layer1_out, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer2_bn = tf.layers.batch_normalization(conv_layer2, training=is_training)
    conv_layer2_out = tf.nn.relu(conv_layer2_bn)
    
    conv_layer3 = tf.layers.conv2d(conv_layer2_out, layer_depth, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    conv_layer3_bn = tf.layers.batch_normalization(conv_layer3, training=is_training)
    conv_layer3_out = tf.nn.relu(conv_layer3_bn)
    pool_layer3 = tf.layers.max_pooling2d(conv_layer3_out,[2,2],strides=2,padding='same')
    return pool_layer3,conv_layer3_out


def vgg16_backbone(layer,is_training):
    end_points={}
    #conv1
    layer,_ = conv_layer_2conv(layer, 64, is_training=is_training)
    #conv2
    layer,_ = conv_layer_2conv(layer, 128, is_training=is_training)
    #conv3
    layer,_ = conv_layer_3conv(layer, 256, is_training=is_training)
    #conv4
    layer,foot_stage1 = conv_layer_3conv(layer, 512, is_training=is_training)
    end_points['block4'] = foot_stage1
    #conv5
    layer = tf.layers.conv2d(layer, 512, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, 512, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.layers.max_pooling2d(layer,[3,3],strides=1,padding='same')
    layer = tf.nn.relu(layer)

    #FC6
    layer = slim.conv2d(layer, 1024, [3, 3], rate=6, scope='conv6')

    #FC7
    layer = tf.layers.conv2d(layer, 1024, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    end_points['block7'] = layer

    #conv8
    layer = tf.layers.conv2d(layer, 256, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, 512, [3,3], 2, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    end_points['block8'] = layer

    #conv9
    layer = tf.layers.conv2d(layer, 128, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, 256, [3,3], 2, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    end_points['block9'] = layer

    #conv10
    layer = tf.layers.conv2d(layer, 128, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, 256, [3,3], 1, 'valid', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    end_points['block10'] = layer

    #conv11
    layer = tf.layers.conv2d(layer, 128, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, 256, [3,3], 1, 'valid', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    end_points['block11'] = layer
    
    return end_points
