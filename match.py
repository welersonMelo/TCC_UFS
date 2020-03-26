import sys
import cv2
import numpy as np
import math
from scipy.spatial import distance
from numpy import linalg as LA

def printLineOnImages(argv, matchedDesc, correspKPList):
    imgPath1 = ""
    imgPath2 = ""
    if len(argv) > 6:
        imgPath1 = argv[5]
        imgPath2 = argv[6]
        img1 = cv2.imread(imgPath1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(imgPath2, cv2.IMREAD_UNCHANGED)
        h = img1.shape[0]
        w = img1.shape[1]
        
        h2 = img2.shape[0]
        w2 = img2.shape[1]
        
        if len(img1.shape) == 2:
            vis = np.zeros((max(h, h2), w+w2), np.uint8)
            vis[:h, :w] = img1
            vis[:h2, w:w+w2] = img2

            cv2.line(vis, (w, 0), (w, h), 50, 1)
        else:
            vis = np.zeros((max(h, h2), w+w2, 3), np.uint8)
            vis[:h, :w, :3] = img1
            vis[:h2, w:w+w2, :3] = img2

            cv2.line(vis, (w, 0), (w, h), (50,50,50), 1)

        for point1, point2 in matchedDesc.items():
            x1, y1 = int(point1[0]), int(point1[1])
            x2, y2 = int(point2[0]) + w, int(point2[1])

            if point1 in correspKPList and correspKPList[point1].all() == point2.all():
                if abs(point1[0]-point2[0]) < 15 and abs(point1[1]-point2[1]) < 15:
                    cv2.line(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            else:
                cv2.line(vis, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imwrite('descriptorResult.png', vis)
    else:
        print('Passe as imagens de entrada na chamada do python para poder usar esta funcao.')

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

def matchedDesc(filePath1, filePath2, isBinario, calcDist, th):
    featVecListRef = []
    featVecListQ = []

    # Reading file with descriptor vector
    featVecListRef = readFromFile(filePath1)
    featVecListQ = readFromFile(filePath2)

    # Threshold to match
    threshold = th

    # Matched Descriptors
    matchedDesc = dict()

    totalSum = 0

    # Find best matches for each descriptor vector, compare 
    for vecRef in featVecListRef:
        distList = []
        _vecRef = np.array(vecRef[2:])

        for vecQ in featVecListQ:
            _vecQ = np.array(vecQ[2:])
            distList.append(calcDist(_vecRef, _vecQ))
        
        biggest = np.max(np.array(distList))

        # Getting index of smallest dist and 2nd smallest
        idSmallest1 = np.argmin(np.array(distList))
        dBM1 = distList[idSmallest1]
        distList[idSmallest1] = 1e9
        idSmallest2 = np.argmin(np.array(distList))
        dBM2 = distList[idSmallest2]

        while True:
            if dBM1 == dBM2:
                distList[idSmallest2] = 1e9
                idSmallest2 = np.argmin(np.array(distList))
                dBM2 = distList[idSmallest2]
            else:
                break

        # Comparing threshold and saving the match
        ratio = 1.0*dBM1/max(1e-6, dBM2)

        if ratio < threshold:
            matchedDesc[(vecRef[0], vecRef[1])] = np.array([featVecListQ[idSmallest1][0], featVecListQ[idSmallest1][1]])
        
    return matchedDesc

def getCorrespondentPoints(filePath1, filePath2, calcDist, hMatrix, error):
    featVecListRef = readFromFile(filePath1)
    featVecListQ = readFromFile(filePath2)

    correspKPList = dict()

    for vecRef in featVecListRef:
        _vecR = np.array([vecRef[0], vecRef[1]])
        minor = 1e9
        correspKP = None
        for vecQ in featVecListQ:
            _vecQ = np.array([vecQ[0], vecQ[1]])

            vecAux = hMatrix.dot(np.array([_vecQ[0], _vecQ[1], 1.0]).transpose())
            vecAux = np.array([vecAux[0]/vecAux[2], vecAux[1]/vecAux[2]]) 
            
            #input('...')

            d = calcDist(vecAux, _vecR)

            if d < error:
                if d < minor:
                    minor = d
                    correspKP = _vecQ
                    #print(vecAux,':',_vecR)

        if minor != 1e9:
            correspKPList[(_vecR[0], _vecR[1])] = correspKP
            df1 = abs(_vecR[0]-correspKP[0])
            df2 = abs(_vecR[1]-correspKP[1])
            if df1 > 8 or df2 > 8:
                print(df1, df2)
    
    return correspKPList

#####################################################
### How to call    agrs: 1-Filepath to the first image(ref); 2-Filepath to the second image(Query); 3-is a Binary Descriptor:1 or 0; 4- Homography matrix path
# python3 match.py ../TestImages/imagename.jpg.feats ../TestImages/imagename2.jpg.feats 0 /home/welerson/Área de Trabalho/Pesquisa /dataset/2D/lighting/H.001.011.txt
#####################################################

####################
####### main #######
####################

filePath1 = sys.argv[1]
filePath2 = sys.argv[2]
isBinario = int(sys.argv[3])
hMatrixPath = sys.argv[4]

calcDist = None
# Passing function do calc dist according to the descriptor type
if isBinario:
    calcDist = calcDistHamming
else:
    calcDist = calcDistEuclidian


# homagraphy matrix
hMatrix = np.array(readFromFile(hMatrixPath))

# error ratio for kp repeatability calc
error = 10.0

correspKPList = getCorrespondentPoints(filePath1, filePath2, calcDist, hMatrix, error)

### TESTE PARA RESULTADO TCC ###
'''
featVecListRef = readFromFile(filePath1)
featVecListQ = readFromFile(filePath2)

for key, point in correspKPList.items():
    print (key, '->', point)

    # para cada key (x,y) no dic, encontrar o descritor em (x,y) correspondente com menor distancia
    # 

thStep = 1e-1 #step to increase threshold
threshold = 1.1

sumS, sumB, totalSum = matchedDesc(filePath1, filePath2, isBinario, calcDist, threshold)

print (sumS, 1.0*sumS/totalSum, ':', sumB, sumB/totalSum)

exit()
### FIM DO TESTE ###
'''

thStep = 0.05 #step to increase threshold
threshold = 0.7

precision = []
recall = []

#print ('corr List:', len(correspKPList))

while threshold <= 1.0:
    matchKPList = matchedDesc(filePath1, filePath2, isBinario, calcDist, threshold)
    
    #Print lines on an output image
    printLineOnImages(sys.argv, matchKPList, correspKPList)
    exit()

    contTruePosi = 0
    for key, point in correspKPList.items():
        if key in matchKPList and matchKPList[key].all() == point.all():
            contTruePosi += 1

    recall.append(contTruePosi/max( len(correspKPList), 0.0000001 ) )
    if len(matchKPList) == 0:
        precision.append(1)
    else:
        precision.append(contTruePosi/len(matchKPList))
    
    threshold+= thStep

print ('r:', recall)
print ('p:', precision)

avgPrecision = 0
n = len(recall)
for i in range(1, n):
    avgPrecision += (recall[i] - recall[i-1]) * precision[i]

print(avgPrecision)
