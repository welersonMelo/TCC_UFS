class KeyPoint:
    def __init__(self, x, y, dir):
        self.x = x
        self.y = y
        self.dir = dir

    def setDir(self, dir):
        self.dir = dir

class KeyPointList:
    ## Make sure that absPath ends with '/' and subpath does not 
    def __init__(self, absPath, subpath, typeNAlg):
        auxList = []
        completePath = absPath+subpath + "/" + subpath + "." + typeNAlg
        
        first = True
        f = open(completePath + "1.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 0)
            auxList.append(kp)
        f.close()

        first = True
        f = open(completePath + "2.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 0)
            auxList.append(kp)
        f.close()

        first = True
        f = open(completePath + "3.txt", "r")
        for line in f:
            if first:
                first = False
                continue
            
            y, x, v = line.split()

            kp = KeyPoint(int(x), int(y), 0)
            auxList.append(kp)
        f.close()

        self.List = auxList

#keypoints = KeyPointList('/home/welerson/Área de Trabalho/Pesquisa /dataset/2D/distance/', '100', 'HDR.surf')
