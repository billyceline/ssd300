import numpy as np
import cv2

def data_preprocessing(image_batch,annotation_batch,cls_batch,target_size):

    x_batch_list =list()
    y_batch_list =list()
    w_batch_list =list()
    h_batch_list =list()
    cls_batch_list = list()
    image_batch_list = list()
    anno_batch_list = list()
    for i in range(len(annotation_batch)):
        #loop for each img
        temp_anno = list()
        temp_cls = list()
        image_batch_temp,xmin,ymin,xmax,ymax,_=resize_img_bbox(image_batch[i],target_size,annotation_batch[i])
        
        image_batch_list.append(image_batch_temp)
        for j in range(len(annotation_batch[i])):
            #loop for each bbox in one img
            temp_anno.append([ymin[j]/300,xmin[j]/300,ymax[j]/300,xmax[j]/300])
            temp_cls.append(int(cls_batch[i][j]))
        for j in range(60-len(annotation_batch[i])):
            temp_anno.append([0,0,0,0])
            temp_cls.append(0)
        anno_batch_list.append(np.array(temp_anno,dtype=np.float32))
        cls_batch_list.append(np.array(temp_cls,dtype=np.int32))
        
    image_batch = np.array(image_batch_list,dtype=np.float32)
#     anno_batch_list = np.array(anno_batch_list,dtype=np.float32)
#     cls_batch_list = np.array(cls_batch_list,dtype=np.int32)
    return image_batch,anno_batch_list,cls_batch_list

def resize_img_bbox(original_img,target_size,annotation):
    xmin_resized = []
    ymin_resized = []
    xmax_resized = []
    ymax_resized = []
    ratio_list = []
    img = cv2.resize(original_img,(target_size,target_size))
    for i in annotation:
        xmin = int(float(i[0]))
        ymin = int(float(i[1]))
        xmax = int(i[2])
        ymax = int(i[3])

        x_ratio = 300/original_img.shape[1]
        y_ratio = 300/original_img.shape[0]
        ratio_list.append([x_ratio,y_ratio])
        xmin_resized.append(int(np.round(xmin*x_ratio)))
        ymin_resized.append(int(np.round(ymin*y_ratio)))
        xmax_resized.append(int(np.round(xmax*x_ratio)))
        ymax_resized.append(int(np.round(ymax*y_ratio)))
    return img,xmin_resized,ymin_resized,xmax_resized,ymax_resized,ratio_list

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
