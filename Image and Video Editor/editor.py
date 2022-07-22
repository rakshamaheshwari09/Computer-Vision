# Importing required packages
import cv2
import numpy as np
import os
import sys

# Parsing command line arguments
img_arg = sys.argv[1]
path = 'assets/images/' + str(img_arg)

if len(sys.argv)>2:
    height, width = int(sys.argv[3]), int(sys.argv[2])
else:
    height, width = 720,1280

# Checking for image file
if(os.path.exists(path)):
    img = cv2.imread(path)
else:
    img = np.zeros((512,512,3))
    img.fill(255)

img = cv2.resize(img, (width,height))

# Defining functions to be called in setMouseCallback
red = np.array([0,0,255])
blue = np.array([255,0,0])
green = np.array([0,255,0])

def mousePointsred(event, x, y, flags, params):
    global coord
    if event == cv2.EVENT_LBUTTONDOWN:
        img[y,x] = red
        cv2.imshow("Editor", img)

def mousePointsblue(event, x, y, flags, params):
    global coord
    if event == cv2.EVENT_LBUTTONDOWN:
        img[y,x] = blue
        cv2.imshow("Editor", img)


def mousePointsgreen(event, x, y, flags, params):
    global coord
    if event == cv2.EVENT_LBUTTONDOWN:
        img[y,x] = green
        cv2.imshow("Editor", img)

# Displaying the image in a loop, with appropriate key events
while True:
    cv2.namedWindow("Editor")
    cv2.moveWindow("Editor",-12,-10)
    cv2.imshow("Editor", img)

    # The function waitKey waits for a key event infinitely (when delay<=0)
    k = cv2.waitKey(100)

    if k == ord('r'):     
        print('red')
        cv2.setMouseCallback("Editor", mousePointsred)    
    elif k == ord('g'):
        print('green')
        cv2.setMouseCallback("Editor", mousePointsgreen)   
    elif k == ord('b'):  #escape key 
        print('blue')
        cv2.setMouseCallback("Editor", mousePointsblue)   
    elif k == ord('q'):
        cv2.imwrite('assets/images/edit.jpg',img)
        break



