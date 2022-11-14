from imutils.perspective import four_point_transform
import cv2
import numpy
import os

# Load image, grayscale, Gaussian blur, Otsu’s threshold
p1 = "/root/medical-img/dataset/origin/"
(dir_path, dir_names, filenames) = next(os.walk(os.path.abspath(p1)))
for fname in filenames:
    imfile = p1 + fname
    print(imfile)
    image = cv2.imread(imfile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = cnts[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    cv2.imwrite("/root/medical-img/dataset/origin_clipped/cropped_" + os.path.basename(imfile),crop)


