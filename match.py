import sys
import cv2
import numpy as np
import math
from scipy.spatial import distance

def readFromFile(filePath):
    listAux = []
    f = open(filePath, "r")
    for line in f:
        x = line.split()
        x = [float(val) for val in x]
        listAux.append(x)
    f.close()
    return listAux

def calcDistEuclidian(vecRef, vecQ):
    return distance.euclidean(vecRef, vecQ)

def calcDistHamming(vecRef, vecQ):
    return 0


#####################################################
### How to call    agrs: 1-Filepath to the first image(ref); 2-Filepath to the second image(Query); 3-is a Binary Descriptor:1 or 0; 
# python3 match.py ../TestImages/imagename.jpg.feats ../TestImages/imagename2.jpg.feats 0 
##### main #####
#####################################################

filePath1 = sys.argv[1]
filePath2 = sys.argv[2]
isBinario = int(sys.argv[3])

imgPath1 = ""
imgPath2 = ""

calcDist = None

# Passing property function do calc dist to a variable
if isBinario:
    calcDist = calcDistHamming
else:
    calcDist = calcDistEuclidian

featVecListRef = []
featVecListQ = []

# Reading file with descriptor vector
featVecListRef = readFromFile(filePath1)
featVecListQ = readFromFile(filePath2)

# Threshold to match
threshold = 1.1

# Matched Descriptors
matchedDesc = []

# Find best matches for each descriptor vector, compare 
for vecRef in featVecListRef:
    distList = []
    _vecRef = np.array(vecRef[2:])

    for vecQ in featVecListQ:
        _vecQ = np.array(vecQ[2:])
        distList.append(calcDist(_vecRef, _vecQ))
    
    # Getting index of smallest dist and 2nd smallest
    idSmallest1 = np.argmin(np.array(distList))
    dBM1 = distList[idSmallest1]
    distList[idSmallest1] = 1e9
    idSmallest2 = np.argmin(np.array(distList))
    dBM2 = distList[idSmallest2]

    # Comparing threshold and saving the match
    ratio = 1.0*dBM1/dBM2
    if ratio < threshold:
        # Saving matched reference x,y point and query x,y point sequentialy in each vector position
        matchedDesc.append([vecRef[0], vecRef[1], featVecListQ[idSmallest1][0], featVecListQ[idSmallest1][1]])

print (matchedDesc)

# Printing lines in the scren

if len(sys.argv) == 6:
    imgPath1 = sys.argv[4]
    imgPath2 = sys.argv[5]
    img1 = cv2.imread(imgPath1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(imgPath2, cv2.IMREAD_UNCHANGED)
    h, w, ch = img1.shape
    h2, w2, ch2 = img2.shape
    vis = np.zeros((max(h, h2), w+w2,3), np.uint8)
    vis[:h, :w,:3] = img1
    vis[:h2, w:w+w2,:3] = img2

    cv2.line(vis, (w, 0), (w, h), (50,50,50), 1)

    for points in matchedDesc:
        x1, y1 = int(points[0]), int(points[1])
        x2, y2 = int(points[2]) + w, int(points[3])

        cv2.line(vis, (x1, y1), (x2, y2), (0,255,0), 1)

    cv2.imwrite('descriptorResult.png', vis)
        
    
