import handleKeypoints
import sys
import cv2
import numpy as np
import math

from numpy import linalg as LA

# consts values
bins = 8
sigma_c = 1.5
radius_c = 3 * sigma_c

# Feature Vector Final List
featVectorList = []

# params: angle in rads, x, y, orige x, orige y, answer axis(0==x,1==y)
def rotateP(o, x, y, ox, oy, ansA):
    ans = 0
    if ansA == 0:
        ans = (x * math.cos(o) - y * math.sin(o)) + ox
    else:
        ans = (x * math.sin(o) + y * math.cos(o)) + oy

    return int(round(ans))

# params: sigma, high of kernel window
def gaussianKernel(sigma, h):
    base = np.ones((h, h, 1))
    kernel = cv2.getGaussianKernel(h, sigma)
    kernel = np.dot(kernel, kernel.transpose())
    return kernel

# checking if is out of bounds for gradient calc
def isOut(img, x, y):
    h, w = img.shape[:2]
    return (x <= 0 or x >= w-1 or y <= 0 or y >= h-1)

##### main #####
#call ex: python3 siftDescriptor.py ../TestImages/100.LDR.surf.kp.txt ../TestImages/100.jpg
kpList = []
# geting keypoints in file
filePath = sys.argv[1]
f = open(filePath, "r")
for line in f:    
    x, y, s, o = line.split()
    kp = handleKeypoints.KeyPoint(int(x), int(y), int(s), int(o))
    kpList.append(kp)
f.close()

imgPath = sys.argv[2]

img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binWidth = int(360/bins)

windowSize = 16
sigma = windowSize/6
kernel = gaussianKernel(sigma, windowSize)
radius = int(windowSize/2)

toRadVal = 180.0/np.pi

hist = [[np.zeros(bins, dtype=np.float32) for _ in range(4)] for _ in range(4)]

for kp in kpList:
    i = 0
    kpDir = kp.dir*10
    theta = kpDir * np.pi/180.0

    for h in range(-radius, radius):
        y = kp.y + h
        oy = y - kp.y
        j = 0
        for w in range(-radius, radius):
            x = kp.x + w
            ox = x - kp.x

            # rotating KP mask 6% of error in rotated mask
            xR = rotateP(theta, ox, oy, kp.x, kp.y, 0)
            yR = rotateP(theta, ox, oy, kp.x, kp.y, 1)
            xA1 = rotateP(theta, ox+1, oy, kp.x, kp.y, 0)
            xS1 = rotateP(theta, ox-1, oy, kp.x, kp.y, 0)
            yA1 = rotateP(theta, ox, oy+1, kp.x, kp.y, 1)
            yS1 = rotateP(theta, ox, oy-1, kp.x, kp.y, 1)

            if isOut(img, xR, yR):
                continue

            dx = int(img[yR, xA1]) - int(img[yR, xS1])
            dy = int(img[yA1, xR]) - int(img[yS1, xR])

            mag = np.sqrt(dx*dx + dy*dy)
            
            # rotating single direction
            angleRotated = (((np.arctan2(dy, dx)+np.pi) * toRadVal) - kpDir) % 360 
            
            binno =  int(angleRotated/binWidth)

            #histInterp = max(1.0 - abs(orientation - (binno*binWidth + binWidth/2))/(binWidth/2), 1e-6)

            vote = mag #* histInterp
            
            #subReg = windowSize/4
            
            if int(i/4) >= 4:
                print ('i:',int(i/4))
            
            if int(j/4) >= 4:
                print ('j:',int(i/4))

            if binno >= 8 or binno < 0:
                print ('binno:',binno)
            
            hist[int(i/4)][int(j/4)][binno] += (kernel[i, j] * vote)
    
            j = j+1
        i = i+1
    
    featVector = np.zeros(bins * windowSize + 2, dtype=np.float32)        
    hist = np.array(hist)
    featVector[2:] = hist.flatten()
    
    featVector /= max(1e-6, LA.norm(featVector))
    featVector[featVector>0.2] = 0.2
    featVector /= max(1e-6, LA.norm(featVector))

    featVector[0] = kp.x
    featVector[1] = kp.y

    featVectorList.append(featVector)

ind = imgPath.rfind('/') + 1
fileName = imgPath[ind:]
with open(fileName+'.feats', 'w') as f:
    for vec in featVectorList:
        out = str(vec).replace('\n', ' ')
        out = out.replace('[', '').replace(']', '')
        f.write(out+"\n")

print ("fim")

## file output format: 
## Each line has a vector descriptor been the first and second position of vector the x and y KP 
## position, respectivly.