import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from tensorflow.python.keras.initializers import he_normal
from set_flags_variables import *


"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding=padding,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9997,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Active(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, bottleneck) :
    return tf.layers.dense(inputs=x, units=bottleneck, name='linear') #, use_bias=False


class ResNeXt():
    def __init__(self, x, training, bottleneck=128):
        
        self.cardinality = 32 #【16，32】num of split
        self.out_dim = 128 #【64，128】
        self.out_dim_milt = 2  #【2，4】
        self.blocks = 2 # res_block = (split + transition)
        
        self.training = training
        self.bottleneck = bottleneck
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[7, 7], stride=2, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Active(x)
            return x

    def transform_layer(self, x, stride, deepth, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=deepth, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Active(x)

            x = conv_layer(x, filter=deepth, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Active(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Active(x)

            return x

    def split_layer(self, input_x, stride, deepth, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(self.cardinality) :
                splits = self.transform_layer(input_x, stride=stride, deepth=deepth, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block, max_pool,padding='same'):
        # split + transform(bottleneck) + transition + merge
        deepth = out_dim // self.cardinality
        out_dim_trans = out_dim * self.out_dim_milt

        if max_pool == True:
            input_x = tf.layers.max_pooling2d(input_x,pool_size=[3,3],strides=2,padding=padding)
        else:
            input_x = tf.layers.average_pooling2d(input_x, pool_size=[2,2], strides=2, padding=padding)
            
        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])
            
            if input_dim == out_dim_trans:
                residual = input_x
            else:
                residual = self.transition_layer(input_x, out_dim=out_dim_trans, scope='residual_layer_'+layer_num+'_'+str(i))
                
            x = self.split_layer(residual, stride=1, deepth=deepth, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim_trans, scope='trans_layer_'+layer_num+'_'+str(i))
            input_x = x + residual
            
        return x


    def Build_ResNext(self, input_x):
        # only cifar10 architecture
        end_points={}
        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=self.out_dim, layer_num='1',res_block=1, max_pool=True)
        x = self.residual_layer(x, out_dim=self.out_dim*2, layer_num='2',res_block=1, max_pool='avg_pool',padding='same')
        end_points['block4'] = x
        x = self.residual_layer(x, out_dim=self.out_dim*4, layer_num='3',res_block=1, max_pool='avg_pool',padding='same')
        end_points['block7'] = x
        x = self.residual_layer(x, out_dim=self.out_dim*8, layer_num='4',res_block=1, max_pool='avg_pool',padding='same')
        end_points['block8'] = x
        x = self.residual_layer(x, out_dim=self.out_dim*8, layer_num='5',res_block=1, max_pool='avg_pool',padding='same')
        end_points['block9'] = x
        x = self.residual_layer(x, out_dim=self.out_dim*8, layer_num='6',res_block=1, max_pool='avg_pool',padding='same')
        end_points['block10'] = x
        x = self.residual_layer(x, out_dim=self.out_dim*8, layer_num='7',res_block=1, max_pool='avg_pool',padding='valid')
        end_points['block11'] = x
        return end_points
