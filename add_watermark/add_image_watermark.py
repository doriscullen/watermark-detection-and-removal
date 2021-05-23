import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont

# number of images
img_num = 3

# the folder where you save the orginal images
path = "../dataset/train/original/"

# the folder where you wish to save cropped images
outpath = "../dataset/train/original/"
size = (300,300)

if not os.path.exists(path):
	warnings.warn("Folder does not exist.", UserWarning)

if not os.path.isdir(outpath):
    os.makedirs(outpath) 

# the folder to save the watermarked images
savePath = "../dataset/train/watermarked/"

if not os.path.isdir(savePath):
    os.makedirs(savePath)

# rename your images as "1.png" "2.png" ...
i = 1
files = []
files = os.listdir(path)
for item in files:
    src = os.path.join(os.path.abspath(path), item)
    dst = os.path.join(os.path.abspath(path), str(i) + '.png')
    img = cv2.imread(src)
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, out)
    print(i)
    i += 1
    if i>img_num: break


wm = Image.open("multiple.png")

for i in range(1,img_num+1):
    image=Image.open(path+str(i)+".png")
    layer=image.convert('RGBA') # convert to RGB
    res1=Image.alpha_composite(layer,wm)
    res1.save(savePath+str(i)+".png") # save picture
    print("success "+str(i))