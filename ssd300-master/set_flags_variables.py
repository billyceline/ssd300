import tensorflow as tf
#set Flags
tf.app.flags.DEFINE_integer('image_size', 300, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size for training.")
tf.app.flags.DEFINE_integer('num_class', 21, "Actual num of class +1.")
tf.app.flags.DEFINE_float('weight_decay', 5e-4, "Weight decay for l2 regularization.")
tf.app.flags.DEFINE_string('log_dir','./SSD_Billy/log','tensorboard directory')
tf.app.flags.DEFINE_string('checkpoint_dir','/media/xinje/New Volume/ssd/vgg16/hnm','The directory where to save the parameters of the network')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')


#set some constant values
feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_size_bounds=[0.15, 0.90]
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
anchor_offset=0.5
normalizations=[20, -1, -1, -1, -1, -1]
prior_scaling=[0.1, 0.1, 0.2, 0.2]
learning_rate = 1e-4
match_threshold = 0.5
score_threshold =0.5
