import tensorflow as tf
from set_flags_variables import *
import numpy as np
import sys
import os
import math
add_dir = os.path.abspath('./data/')
sys.path.append(add_dir)
from data import custom_layers
from data import ssd_common
from tensorflow.python.keras.initializers import he_normal


def ssd_multibox_layer(layer,anchor_sizes,anchor_ratios,feat_shapes,normalization=False):
    #need to figure out why length of anchor_size + length of anchor_ratios
    if normalization > 0:
        layer = custom_layers.l2_normalization(layer, scaling=True)
    num_anchors = len(anchor_sizes)+len(anchor_ratios)
    num_loc_pred = num_anchors *4
    num_cls_pred = num_anchors * FLAGS.num_class
    
    pred = tf.layers.conv2d(layer, num_loc_pred+num_cls_pred, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    loc_pred = tf.reshape(pred[...,0:(4*num_anchors)],[-1,feat_shapes[0],feat_shapes[0], num_anchors,4])
    
#     cls_pred = tf.layers.conv2d(layer, num_cls_pred, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=0.01),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
    cls_pred = tf.reshape(pred[...,(4*num_anchors)::],[-1,feat_shapes[0],feat_shapes[0],num_anchors,FLAGS.num_class])
    #channel to last and reshape didn't implement
    return cls_pred,loc_pred

def ssd_anchor_one_layer(img_shape,#原始图像shape
                         feat_shape,#特征图shape
                         sizes,#预设的box size
                         ratios,#aspect 比例
                         step,#anchor的层
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    
    """
    #测试中，参数如下
    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)]
    anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]]
    anchor_steps=[8, 16, 32, 64, 100, 300]


    offset=0.5

    dtype=np.float32

    feat_shape=feat_shapes[0]
    step=anchor_steps[0]
    """
    #测试中，y和x的shape为（38,38）（38,38）
    #y的值为
    #array([[ 0,  0,  0, ...,  0,  0,  0],
     #  [ 1,  1,  1, ...,  1,  1,  1],
    # [ 2,  2,  2, ...,  2,  2,  2],
    #   ..., 
     #  [35, 35, 35, ..., 35, 35, 35],
    #  [36, 36, 36, ..., 36, 36, 36],
     #  [37, 37, 37, ..., 37, 37, 37]])
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    #测试中y=(y+0.5)×8/300,x=(x+0.5)×8/300
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    #扩展维度，维度为（38,38,1）
    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    #数值为2+2
    num_anchors = len(sizes) + len(ratios)
    #shape为（4,）
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    #测试中，h[0]=21/300,w[0]=21/300?
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        #h[1]=sqrt(21*45)/300
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    #测试中，y和x shape为（38,38,1）
    #h和w的shape为（4,）
    return y, x, h, w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def stack_tensor(tensor):
    final_result = []
    for j in range(6):
        temp_tensor = []
        for i in range(FLAGS.batch_size):
            temp_tensor.append(tensor[i][j])
        final_result.append(tf.stack(temp_tensor,axis=0))
    return final_result

def encode_bboxes(labels,bboxes,anchor_layers_original):
    feat_labels = []
    feat_localizations = []
    feat_scores = []
    for i in range(FLAGS.batch_size):
        feat_labels_temp, feat_localizations_temp, feat_scores_temp = ssd_common.tf_ssd_bboxes_encode(labels[i],bboxes[i],anchor_layers_original,FLAGS.num_class,0)
        feat_labels.append(feat_labels_temp)
        feat_localizations.append(feat_localizations_temp)
        feat_scores.append(feat_scores_temp) 

    feat_labels = stack_tensor(feat_labels)
    feat_localizations = stack_tensor(feat_localizations)
    feat_scores = stack_tensor(feat_scores)
    return feat_labels,feat_localizations,feat_scores


def reshape_tensors(localisations_layer,logits_layer,predictions_layer):
    localisations_layer = tf.reshape(localisations_layer,(1,-1,4))
    logits_layer = tf.reshape(logits_layer,(1,-1,21))
    predictions_layer = tf.reshape(predictions_layer,(1,-1,21))
    return localisations_layer,logits_layer,predictions_layer

def inference_network_part(decoded_boxes,logits,predictions):
    reshaped_locals = []
    reshaped_logits = []
    reshaped_predictions = []
    for i in range(len(feat_layers)):
        temp_locals,temp_logits,temp_predictions = reshape_tensors(decoded_boxes[i],logits[i],predictions[i])
        reshaped_locals.append(temp_locals)
        reshaped_logits.append(temp_logits)
        reshaped_predictions.append(temp_predictions)
    reshaped_locals = tf.concat(reshaped_locals,axis=1)
    reshaped_logits = tf.concat(reshaped_logits,axis=1)
    reshaped_predictions = tf.concat(reshaped_predictions,axis=1)

    classes = tf.cast(tf.argmax(reshaped_predictions,axis=2),tf.int32)
    scores = tf.reduce_max(reshaped_predictions,axis=2)
    #remove boxes belongs to background
    scores = scores * tf.cast(classes >0, scores.dtype)
    #remove boxes with low scores
    mask = tf.greater(scores, 0.5)
    classes = classes * tf.cast(mask, classes.dtype)
    scores = scores * tf.cast(mask, scores.dtype)
    ymin = tf.zeros_like(reshaped_locals[...,0])
    xmin = tf.zeros_like(reshaped_locals[...,1])
    ymax = tf.ones_like(reshaped_locals[...,2])
    xmax = tf.ones_like(reshaped_locals[...,3])

    ymin = tf.maximum(reshaped_locals[...,0],ymin)
    xmin = tf.maximum(reshaped_locals[...,1],xmin)
    ymax = tf.minimum(reshaped_locals[...,2],ymax)
    xmax = tf.minimum(reshaped_locals[...,3],xmax)
    boxes = tf.stack([ymin,xmin,ymax,xmax],axis = -1)

    #start to remove boxes
    mask_score = tf.greater_equal(scores,0.5)
    mask_class = tf.not_equal(classes, 0)
    total_mask = tf.logical_and(mask_score,mask_class)
    total_index = tf.where(total_mask[0])
    #remove background and scores>0.5
    scores = tf.gather(scores,total_index[:,0],axis=1)
    classes = tf.gather(classes,total_index[:,0],axis=1)
    boxes = tf.gather(boxes,total_index[:,0],axis=1)
    nms_index = tf.image.non_max_suppression(boxes[0], scores[0],50, iou_threshold=0.5)
    #remove nms boxes
    scores = tf.gather(scores[0],nms_index,name='scores')
    classes = tf.gather(classes[0],nms_index,name='classes')
    boxes = tf.gather(boxes[0],nms_index,name='boxes')
