import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import os

# path = "../dataset/origin/"
# img_resized_dir = "../dataset/origin_clipped/"
path = "../dataset/test/test_D/"
img_resized_dir = "../dataset/test_cropped/test_D/"
dirs = os.listdir(path)

def thyroid_scale():
    a = 0
    for item in dirs:
        if os.path.isfile(path+item):
            print(item)
            a += 1
            img = cv2.imread(path+item)
            img = cv2.resize(img, (512,512))
            img = img[12:500, 12:500]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (11, 11), 1)
            # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ret,thresh = cv2.threshold(gray, 1, 255, 0)
            
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            areas = [cv2.contourArea(c) for c in contours]
            
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            crop_img = img[y+35:y+h-5,x+25:x+w-10]
            # resize_img = cv2.resize(crop_img, (300, 250), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(img_resized_dir+item, crop_img)
            print(a)
thyroid_scale() 