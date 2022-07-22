import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def histeq(img):
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))

    return equ

feature_extractor = 'orb'
feature_matching = 'bf'

train_img = cv2.imread('assets/campus/campus1.jpg')
train_img = histeq(train_img)
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

query_img = cv2.imread('assets/campus/campus2.jpg')
query_img = histeq(query_img)
query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout = False, figsize = (16,9))
ax1.imshow(query_img, cmap="gray")
ax1.set_xlabel("Query Image", fontsize=14)
ax2.imshow(train_img, cmap="gray")
ax2.set_xlabel("Train Image", fontsize=14)

plt.show()

def Detect(image, method=None):
    descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)

kps1, features1 = Detect(train_img_gray, method = feature_extractor)
kps2, features2 = Detect(query_img_gray, method = feature_extractor)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(train_img_gray, kps1, None, color=(0,255,0)))
ax1.set_xlabel("1", fontsize = 14)
ax2.imshow(cv2.drawKeypoints(query_img_gray, kps2, None, color=(0,255,0)))
ax2.set_xlabel("2", fontsize = 14)

plt.show()

def Match(method, crossCheck):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def MatchPoints(features1, features2, method):
    bf = Match(method, crossCheck = True)
    best_matches = bf.match(features1, features2)

    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))

    return rawMatches

fig = plt.figure(figsize = (20,8))
matches = MatchPoints(features1, features2, method = feature_extractor)
img3 = cv2.drawMatches(train_img, kps1, query_img, kps2, matches[:100], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()

def Homography(kps1, kps2, features1, features2, matches, reprojThresh):
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])
    
    if len(matches) > 4:
        ptsA = np.float32([kps1[m.queryIdx] for m in matches])
        ptsB = np.float32([kps2[m.trainIdx] for m in matches])
        
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

M = Homography(kps1, kps2, features1, features2, matches, reprojThresh=4)
if M is None:
    print("Error!")
(matches, H, status) = M
print(H)

width = train_img.shape[1] + query_img.shape[1]
height = train_img.shape[0] + query_img.shape[0]

result = cv2.warpPerspective(train_img, H, (width, height))
result[0:query_img.shape[0], 0:query_img.shape[1]] = query_img

plt.figure(figsize=(20,10))
plt.imshow(result)

plt.axis('off')
plt.show()

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key = cv2.contourArea)
(x,y,w,h) = cv2.boundingRect(c)

result = result[y:y+h, x:x+w]

plt.figure(figsize=(20,10))
plt.imshow(result)
plt.axis('off')
plt.show()

cv2.imwrite('assets/campus/output.jpg', result)