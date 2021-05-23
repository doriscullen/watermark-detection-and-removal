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

# add watermark    
for i in range(1, img_num+1):
    image=Image.open(path+str(i)+".png")
    layer=image.convert('RGBA') # convert to RGB
    text='watermark'            # content of the watermark
    text_font=ImageFont.truetype("Antonio-Regular.ttf", 30) # set font and text size
    text_overlay=Image.new('RGBA',layer.size,(255,255,255,0))   # create a layer
    # create watermark layer
    image_draw=ImageDraw.Draw(text_overlay)
    text_size_x,text_size_y=image_draw.textsize(text,text_font) # get watermark size
    
    # add watermark at random position
    pos_x=random.randint(0,layer.size[0]-text_size_x)
    pos_y=random.randint(0,layer.size[1]-text_size_y)
    text_xy=(int(pos_x), int(pos_y))    # get position
    image_draw.text(text_xy,text,font=text_font,fill=(255,255,255,100)) # set color, alpha, and position
    res2=Image.alpha_composite(layer, text_overlay)
    res2.save(savePath+str(i)+"_random.png") # save picture
    print("success "+str(i))