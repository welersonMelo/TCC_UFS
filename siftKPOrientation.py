import handleKeypoints
import sys
import cv2
import numpy as np
import math

# consts values
bins = 36
sigma_c = 1.5
radius_c = 3 * sigma_c

def gaussianKernel(sigma, h):
    base = np.ones((h, h, 1))
    kernel = cv2.getGaussianKernel(h, sigma)
    kernel = np.dot(kernel, kernel.transpose())
    return kernel

# checking if is out of bounds for gradient calc
def isOut(img, x, y):
    h, w = img.shape[:2]
    return (x <= 0 or x >= w-1 or y <= 0 or y >= h-1)

### function that gets the orientation of a passed KP of Keypoint type
def calcOrientation(img, kp):
    sigma = sigma_c * kp.scale
    radius = int(radius_c * kp.scale)
    hist = np.zeros(bins, dtype=np.float32) 

    kernel = gaussianKernel(sigma, radius*2+1)

    for i in range(-radius, radius+1):
        y = kp.y + i
        if isOut(img, 1, y):
            continue
        for j in range(-radius, radius+1):
            x = kp.x + j
            if isOut(img, x, 1):
                continue
            dx = int(img[y, x+1]) - int(img[y, x-1])
            dy = int(img[y+1, x]) - int(img[y-1, x])

            mag = np.sqrt(dx*dx + dy*dy)
            
            orientation = (np.arctan2(dy, dx)+np.pi) * 180/np.pi
            
            hist[int(np.floor(orientation)/10)-1] += (kernel[i, j] * mag)
        
        maxBin = [np.argmax(hist)]
        maxBinVal = hist[maxBin[0]]
        # checking if exist other valeus above 80% of the max
        for k in range(len(hist)):
            if k == maxBin[0]:
                continue
            if hist[k] >= .8*maxBinVal:
                maxBin.append(k)
        
        return maxBin


##### main #####
# Ex call:
#python3 siftKPOrientation.py ../TestImages/100.jpg /home/welerson/Área\ de\ Trabalho/Pesquisa\ /dataset/2D/distance/100/ 100.LDR.surf
imgPath = sys.argv[1]
path = sys.argv[2] 
fileName = sys.argv[3]

#imgPath = '../TestImages/home.jpg'
#path = '/home/welerson/Área de Trabalho/TCC/Implementações/TestImages/'
#fileName = 'home.dog'

# read image and passing to gray scale
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gaussian filter input img
img = cv2.GaussianBlur(img, (5, 5), 1.5)

# getting keypoints in file 
keypoints = handleKeypoints.KeyPointList(path, fileName)

# calculating orientation for keypoints in list of keypoints 
auxList = []
for kp in keypoints.List:
    directions = calcOrientation(img, kp)
    kp.setDir(directions[0])

    px = int(30*(np.cos(np.radians(kp.dir*10))))
    py = int(30*(np.sin(np.radians(kp.dir*10))))
    
    #print (kp.dir*10, ':', (kp.x + px, kp.y + py), (kp.x, kp.y))
    cv2.arrowedLine(img, (kp.x, kp.y), (kp.x + px, kp.y + py), (0, 0, 255), 1)
    
    directions.pop(0)
    if len(directions) > 0:
        for d in directions:
            nkp = handleKeypoints.KeyPoint(kp.x, kp.y, kp.scale, d)
            auxList.append(nkp)

# passando os novos KP para a lista final
for newKp in auxList:
    keypoints.List.append(newKp)

#cv2.imwrite('kpOrientation.jpg', img)

with open(fileName+'.kp.txt', 'w') as f:
    for item in keypoints.List:
        f.write(str(item.x)+' '+str(item.y)+' '+str(item.scale)+' '+str(item.dir)+'\n')