# Importing required packages
import cv2
import os
import sys

# Parsing command line arguments
vid_arg = sys.argv[1]
path = 'assets/videos/' +str(vid_arg)

if len(sys.argv)>2:
    height, width = int(sys.argv[3]), int(sys.argv[2])
else:
    height, width = 720,1280

# Webcam video
webcam = cv2.VideoCapture(0)
key=k=96

# Two while loops to ensure that video is looped back after it ends
while True:
    # Variable to check for breaking the loop on command
    isclosed=0
    video = cv2.VideoCapture(path)
    while (True):
        ret2, frame2 = video.read()

        # It should only show the frame when the ret is true
        if ret2 == True:
            resize = cv2.resize(frame2, (width, height))
            l1 = cv2.line(resize,(630,370),(650,350),(255,0,0),4)
            l2 = cv2.line(resize,(630,350),(650,370),(255,0,0),4)
            cv2.namedWindow("Video")
            cv2.moveWindow("Video",-12,-10)
            cv2.imshow("Video",resize)

        else:
            break        


        ret1, frame1 = webcam.read()
        if k == -1:
            pass
        else:
            key = k
        k = cv2.waitKey(1)
        
        if key == 49:
            pass
        elif key == 50:
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame1 = gray
        elif key == 51:
            blur_img = cv2.blur(frame1, (7,7))
            frame1 = blur_img
        elif key == 113:
            isclosed = 1
            break
        
        cv2.namedWindow('Webcam')
        frame1 = cv2.copyMakeBorder(frame1,5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,255])
        resize_webcam = cv2.resize(frame1, (200, 200))
        cv2.imshow('Webcam',resize_webcam)


            
    # To break the loop if Q key is pressed
    if isclosed:
        break
  

# Destroy all windows
video.release()
webcam.release()

cv2.destroyAllWindows()