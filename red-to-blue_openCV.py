import numpy as np
import argparse
import cv2
import imutils
import time


# python red-to-blue.py -i Files/car_red.jpg
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="path to the input image file")
ap.add_argument("-c","--color",help="original color")
ap.add_argument("-n","--name",help="output_name")
args = vars(ap.parse_args())

# car in the HSV color space, then initialize the
redLower1 = (0, 10, 100)
redUpper1 = (10, 255, 255)
redLower2 = (155, 10, 50)
redUpper2 = (180, 255, 255)
blueLower = (90, 50, 50)
blueUpper = (135, 255, 230)

image = cv2.imread(args["image"])
image_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# masks
mask0 = cv2.inRange(image_hsv, redLower1, redUpper1)
mask1 = cv2.inRange(image_hsv, redLower2, redUpper2)
mask_red = mask0+mask1
mask_blue = cv2.inRange(image_hsv, blueLower, blueUpper)


# colors
color = args["color"]
blue = 245
red = 0
if color == 'blue':
    mask = mask_blue
    diff_color = blue - red
    name='car2red.png'
else:
    mask = mask_red
    diff_color = red - blue
    name = 'car2blue.png'

if args["name"]:
    name = args["name"]

name = "Outputs/" + name
# set my output img to zero everywhere except my mask
object = image.copy()
object[np.where(mask == 0)] = 0
object_hsv = image_hsv.copy()
object_hsv[np.where(mask == 0)] = 0
#cv2.imshow('object', object)

# background
background = image.copy()
background[np.where(mask != 0)] = 0
background_hsv = image_hsv.copy()
background_hsv[np.where(mask != 0)] = 0
#cv2.imshow('background', background)

# change object color
h,s,v = cv2.split(object_hsv)
vfunc = np.vectorize(lambda x: np.mod(max(x + diff_color, 360 - x - diff_color), 180))
h_new = vfunc(h).astype(np.uint8)
h_new[np.where(mask == 0)] = 0

# recombine channels
object_new = cv2.merge([h_new,s,v])
bgr_object_new = cv2.cvtColor(object_new, cv2.COLOR_HSV2BGR)
#cv2.imshow('object_new', bgr_object_new)

#restore the image
hsv_new = object_new.copy()
hsv_new[np.where(mask == 0)] = background_hsv[np.where(mask == 0)].copy()

bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
cv2.imwrite(name, bgr_new)
cv2.imshow("original", image)
cv2.imshow(name, bgr_new)

cv2.waitKey(0)
cv2.destroyAllWindows()