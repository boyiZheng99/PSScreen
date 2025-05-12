import cv2
import os
import numpy as np



img_dir = '../retinal_dataset/data/train/DDR_trainset'
output_img_dir = '../retinal_dataset/data/train/DDR_trainset'


os.makedirs(output_img_dir,exist_ok=True)


for file in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,file))
    h,w = img.shape[:2]
    if h != w:
        top,bottom,left,right = 0, 0, 0, 0
        if h > w:
            left = (h-w)//2
            right = (h-w)//2
            if (h-w)%2 != 0:
                right += 1
        else:
            top = (w-h)//2
            bottom = (w-h)//2
            if (w-h)%2 != 0:
                bottom += 1
        img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    cv2.imwrite(os.path.join(output_img_dir,file),img)
