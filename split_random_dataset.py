import random
import os
#import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torch
import numpy as np
grayscale_list = os.listdir('~/akshat/dip_project/dip/grayscale_dataset')
median_list = os.listdir('~/akshat/dip_project/dip/mf_dataset')

image_path =  '~/akshat/dip_project/dip/mf_dataset'

mask_path = '~/akshat/dip_project/dip/grayscale_dataset'

list_zip = []

for img in median_list:
    for mask in grayscale_list:
        if img == mask:
            list_zip.append((img,mask))
        
l = len(list_zip)

train_len = int(l*0.7)
val_len = int(l*0.1)
test_len = l-(train_len+val_len)

random.shuffle(list_zip)

train_path = '~/akshat/dip_project/dip/train'
validation_path = '~/akshat/dip_project/dip/validation'
test_path = '~/akshat/dip_project/dip/test'

def normalize(image):
    m = np.max(image)
    n = np.min(image)
    image = (image-n)/(m-n)
    return image

for i in range(train_len):
    image = cv2.imread(image_path+'/'+list_zip[i][0],0)
    mask = cv2.imread(mask_path+'/'+list_zip[i][0],0)
    image=normalize(image)
    mask= normalize(mask)
    image= torch.tensor(image,dtype=torch.float16)
    mask = torch.tensor(mask,dtype=torch.float16)
    new_name = list_zip[i][0].split('.')[0]+'.pt'
    torch.save(image,train_path+'/'+'Images'+'/'+new_name)
    torch.save(mask,train_path+'/'+'Masks'+'/'+new_name)
#    plt.axis('off')
#    plt.imsave(train_path+'/'+'Images'+'/'+list_zip[i][0],image,cmap='gray')
#    plt.axis('off')
#    plt.imsave(train_path+'/'+'Masks'+'/'+list_zip[i][0],mask,cmap='gray')
for i in range(train_len,train_len+val_len):
    image = cv2.imread(image_path+'/'+list_zip[i][0],0)
    mask = cv2.imread(mask_path+'/'+list_zip[i][0],0)
    image=normalize(image)
    mask= normalize(mask)
    image= torch.tensor(image,dtype=torch.float16)
    mask = torch.tensor(mask,dtype=torch.float16)
    new_name = list_zip[i][0].split('.')[0]+'.pt'
    torch.save(image,validation_path+'/'+'Images'+'/'+new_name)
    torch.save(mask,validation_path+'/'+'Masks'+'/'+new_name)
#    plt.axis('off')
#    plt.imsave(validation_path+'/'+'Images'+'/'+list_zip[i][0],image,cmap='gray')
#    plt.axis('off')
#    plt.imsave(validation_path+'/'+'Masks'+'/'+list_zip[i][0],mask,cmap='gray')
for i in range(train_len+val_len,l):
    image = cv2.imread(image_path+'/'+list_zip[i][0],0)
    mask = cv2.imread(mask_path+'/'+list_zip[i][0],0)
    image=normalize(image)
    mask= normalize(mask)
    image= torch.tensor(image,dtype=torch.float16)
    mask = torch.tensor(mask,dtype=torch.float16)
    new_name = list_zip[i][0].split('.')[0]+'.pt'
    torch.save(image,test_path+'/'+'Images'+'/'+new_name)
    torch.save(mask,test_path+'/'+'Masks'+'/'+new_name)
#    plt.imsave(test_path+'/'+'Images'+'/'+list_zip[i][0],image,cmap='gray')
#    plt.axis('off')
#    plt.imsave(test_path+'/'+'Masks'+'/'+list_zip[i][0],mask,cmap='gray')    
#    plt.axis('off')
