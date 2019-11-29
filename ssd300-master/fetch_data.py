import cv2
import numpy as np
import random
from data_aug import data_augmentation

def read_data(img_list,annotation,batch_size,aug):
    num_batch = len(img_list)/batch_size
    count=0
    while(True):
        image_data = []
        annotation_data = []
        for i in range(batch_size):
            temp_index = i+count*batch_size
            temp_index %=len(img_list) 
            image = cv2.imread(img_list[temp_index])
            if aug:
                image = data_augmentation(image)
            image = image[:,:,::-1]
            image = image.astype(np.float32)
            
#             image = cv2.resize(image,(FLAGS.image_size,FLAGS.image_size))
            image = image/255
            image_data.append(image)
            annotation_data.append(annotation[temp_index])
        count+=1
#         image_data = np.array(image_data)
        yield image_data,annotation_data

def get_path_and_annotation(file_path):
    annotation = []
    img_path = []
    line_list = []
    with open(file_path,'r') as f:
        for line in f:
            temp=[]
            line = line.strip('\n')
            line_list.append(line)
        random.shuffle(line_list)

    for i in line_list:
        line = i.split(' ')
        img_path.append(line[0]+' '+line[1])
        temp = []
        temp_inner= []
        for j in range(2,len(line)):
            temp.append(line[j].split(','))
        annotation.append(temp)
    return img_path,annotation
