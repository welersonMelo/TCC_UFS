from scipy.spatial import distance
import numpy as np
import math
import cv2

def getInterc():
    filePath = '../KPS_for_intercection/'

    imageKPs = [[] for _ in range(6)]

    for i in range(6):
        f = open(filePath+str(i+1)+'.txt', "r")
        for line in f:
            x, y, s = line.split()
            imageKPs[i].append(np.array([int(x), int(y)]))
        f.close()

    error = 15.0

    intercList = []

    for p1 in imageKPs[0]:
        contFound = 0
        for i in range(1, 6):
            for p2 in imageKPs[i]:
                if distance.euclidean(p1, p2) < error:
                    contFound += 1
                    break
        #print (contFound)
        if contFound >= 4:
            intercList.append(p1)
    return intercList

intercList = []

intercList = getInterc()
for x in intercList:
    print (x[0], x[1])
exit()

f = open('../KPS_for_intercection/ListaKP_interc', "r")
for line in f:
    x, y = line.split()
    intercList.append(np.array([int(x), int(y)]))
f.close()

img = cv2.imread('../KPS_for_intercection/100.LDR.jpg', cv2.IMREAD_COLOR) 

radius = 5
color = (0,0,255) 
thickness = 5

for p in intercList:
    center_coordinates = (p[1], p[0])
    img = cv2.circle(img, center_coordinates, radius, color, thickness) 
    #print (p[0],',',p[1])

cv2.imwrite('../KPS_for_intercection/interKPs.jpg', img)