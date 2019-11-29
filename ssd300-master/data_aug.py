import cv2
import random
import numpy

def random_flip(image):

    rand_num = random.random()*2-1
    if(rand_num <(-1/3)):
        return cv2.flip(image, -1)
    elif(rand_num>(-1/3) and rand_num < (1/3)):
        return cv2.flip(image, 0)
    elif(rand_num>(1/3)):
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, rand_num)
    
def random_crop(image):
    rate = (random.random()+1)*0.1*0.5 #random crop 10% to 20%
    cropImg = image[int(image.shape[0]*rate):int(image.shape[0]*(1-rate)),int(image.shape[1]*rate):int(image.shape[1]*(1-rate))]
    return cropImg
    
def random_rotate_image(image):
    #random rotate
    (h,w) = image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center,random.random()*360,1.0)
    image = cv2.warpAffine(image,M,(w,h),borderMode = cv2.BORDER_REPLICATE)
    return image

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1. / scale

    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180
    
    # avoid overflow when astype('uint8')
    image[...] = np.clip(image[...], 0, 255)
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def data_augmentation(image):
    image = random_crop(image)
    image = random_flip(image)
    image = random_rotate_image(image)
    image = random_distort_image(image, hue=18, saturation=1.5, exposure=1.5)
    return image

def split_cls_anno(annotation_batch):
    temp_anno=[]
    temp_cls = []
    for i in annotation_batch:
        temp_anno1=[]
        temp_cls1=[]
        for j in i:
            temp_anno1.append([int(j[0]),int(j[1]),int(j[2]),int(j[3])])#xmin,ymin,xmax,ymax
#             temp_anno1.append([int(j[0]),int(j[1]),(int(j[2])-int(j[0])),(int(j[3])-int(j[1]))])
            temp_cls1.append(int(j[4]))
        temp_anno.append(temp_anno1)
        temp_cls.append(temp_cls1)
    return temp_anno,temp_cls

# def data_augmentation(image_batch,annotation_batch,cls_batch): 
#     ###convert xmin,ymin,xmax,ymax to xleft,yleft,w,h to be compatible with tensorlayer format
#     temp_anno = []
#     for i in annotation_batch:
#         temp_anno1=[]
#         for j in i:
#             temp_anno1.append([int(j[0]),int(j[1]),(int(j[2])-int(j[0])),(int(j[3])-int(j[1]))])
#         temp_anno.append(temp_anno1)
        
#     temp_img2 = []
#     temp_anno2 = []
#     temp_cls2 = []
#     for i in range(FLAGS.batch_size):
#         temp_img1,temp_cls1,temp_anno1 = tl.prepro.obj_box_shift(image_batch[i], cls_batch[i],coords=temp_anno[i],fill_mode='constant',is_rescale=False)
#         temp_img1,temp_cls1,temp_anno1 = tl.prepro.obj_box_zoom(temp_img1,temp_cls1,coords=temp_anno1,fill_mode='constant',is_rescale=False)
#         temp_img2.append(temp_img1)
#         temp_anno2.append(temp_anno1)
#         temp_cls2.append(temp_cls1)

#     ##convert x,y,w,h to xmin,ymin,xmax,ymax
#     temp_anno = []
#     for i in temp_anno2:
#         temp_anno1=[]
#         for j in i:
#             temp_anno1.append([j[0],j[1],j[2]+j[0],j[3]+j[1]])
#         temp_anno.append(temp_anno1)
#     return temp_img2,temp_anno,temp_cls2


