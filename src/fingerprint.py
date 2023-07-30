import numpy as np
import math

class Fingerprint:
    def __init__(self, label, type, num_deltas, deltas, num_cores, cores, minutiaes):
        self.label = label
        self.type = type
        self.num_deltas = num_deltas
        self.deltas = deltas
        self.num_cores = num_cores
        self.cores = cores
        self.minutiaes = minutiaes

def __difference(diff):
    if (diff >= -math.pi/2 and diff <= math.pi/2):
        return diff
    elif diff > math.pi/2:
        return diff - math.pi
    elif diff < -math.pi/2:
        return diff + math.pi

def singular_point(o_cortado, img):
    image = img.copy()
    w = 11
    o_cortado = np.reshape(o_cortado, (int(300/w+1), int(300/w+1)))
    poincare_index = np.zeros(o_cortado.shape)
    for i in np.arange(0,o_cortado.shape[0]-1):
        for j in np.arange(0,o_cortado.shape[1]-1):
            # aux = o_cortado[i:i+3,j:j+3]
            if img[i*w,j*w] != 255: #bloco "presta"
                poincare_index[i,j]  = __difference(o_cortado[i-1,j-1] - o_cortado[i-1,j])
                poincare_index[i,j] += __difference(o_cortado[i-1,j] - o_cortado[i-1,j+1])
                poincare_index[i,j] += __difference(o_cortado[i-1,j+1] - o_cortado[i,j+1])
                poincare_index[i,j] += __difference(o_cortado[i,j+1] - o_cortado[i+1,j+1])
                poincare_index[i,j] += __difference(o_cortado[i+1,j+1] - o_cortado[i+1,j])
                poincare_index[i,j] += __difference(o_cortado[i+1,j] - o_cortado[i+1,j-1])
                poincare_index[i,j] += __difference(o_cortado[i+1,j-1] - o_cortado[i,j-1])
                poincare_index[i,j] += __difference(o_cortado[i,j-1] - o_cortado[i-1,j-1])

    cores = (poincare_index < np.radians(-170)) & (poincare_index > np.radians(-190)) * 1
    point_core = np.argwhere(cores) #encontrar em cores, onde tem um valor diferente de 0, ou seja, onde tem core
    if (point_core.all()):
        for i in range(len(point_core)): #marcar core
            point = point_core[i] #point tem 2 dimensões
            image[point[0]*w-4:point[0]*w+4, point[1]*w-4:point[1]*w+4] = 0
            # cv2.imshow("kk", image)
            # cv2.waitKey()
        print (str(len(point_core))+" cores encontrados")

    deltas = (poincare_index > np.radians(170)) & (poincare_index < np.radians(190)) * 1
    point_delta = np.argwhere(deltas) #encontrar em cores, onde tem um valor diferente de 0, ou seja, onde tem core
    if (point_delta.all()):
        for i in range(len(point_delta)):
            point = point_delta[i] #point tem 2 dimensões
            image[point[0]*w-4:point[0]*w+4, point[1]*w-4:point[1]*w+4] = 0
            # cv2.imshow("kk", image)
            # cv2.waitKey()
        print (str(len(point_delta))+" deltas encontrados")

    #contar numero de cores
    num_cores = np.count_nonzero(cores)
    num_deltas = np.count_nonzero(deltas)
    # print num_cores, num_deltas
    if (num_deltas >= 0) and (num_cores == 0):
        return 'arch', point_delta, point_core
    elif num_cores >= 2:
        return 'whorl', point_delta, point_core
    elif (num_deltas == 1) and (num_cores == 1):
        if deltas[0][1] > cores[0][1]:
            return 'left_loop', point_delta, point_core
        else:
            return 'right_loop', point_delta, point_core
    else:
        return 'undefined', point_delta, point_core
    
def remove_spurious(minutiaes):
    new_minutiaes = minutiaes
    #distancia euclidiana das minucias
    # print new_minutiaes
    i = 0
    while (i < len(new_minutiaes)):
        x, y = new_minutiaes[i]

        for j in range(i+1, len(new_minutiaes)):
            l, c = new_minutiaes[j]
            distance = math.sqrt(math.pow(x - l,2) + math.pow(y - c,2))
            if distance < 8: #não são minucias, remove
                new_minutiaes.remove([x,y])
                new_minutiaes.remove([l,c])
                i = -1
                break
        i+=1

    return new_minutiaes

def get_minutiae( img):
    minutiaes = []
    for k in range(1,299): #desconsidera bordas
        for j in range(1,299):
            nb = 0
            submatrix = img[k-1:k+2,j-1:j+2]
            if submatrix[1,1] == 0: #detecta a minúcia no ponto central (j,k)
                for l in range(0,3):
                    for c in range(0,3):
                        if submatrix[l,c] == 0:
                            nb += 1 #conta os pixels pretos da submatriz 3x3

                if nb == 2: #end point
                    minutiaes.append([j,k])
                elif nb == 4: #bifurcation
                    minutiaes.append([j,k])
                elif nb == 5: #crossing
                    minutiaes.append([j,k])
                #else is a isolated point, edge point or is not a minutiae

    return minutiaes