#!/usr/bin/python
# -*- coding: latin-1 -*-


# Import the required modules
import cv2, os
import sys
import numpy as np
import scipy as sp
import math
from PIL import Image
from numpy import linalg as LA
from skimage.morphology import skeletonize
from skimage import data

class Dataset:
    def __init__(self):
        self.paths = []
        self.images = []
        self.skeleton = []
        self.testes = []

    #funções comuns entre os datasets
    def get_images_and_labels(self, dataset_path):
        for image_path in os.listdir(dataset_path):
            if image_path.endswith('.raw'):
                path = os.path.join(dataset_path, image_path)
                self.paths.append(path)

                #image = np.empty((300,300), np.uint8)
                path = open(path, 'rb')
                #image.data[:] = open(path).read()
                image = np.fromfile(path, dtype='uint8', count=300*300)
                image = image.reshape((300, 300))
                path.close()
                self.images.append(image)
                #cv2.imshow("Adding fingerprints to traning set...", image)
                #cv2.waitKey(50)

class Teste:
    def __init__(self, label, type, num_deltas, deltas, num_cores, cores, minutiaes):
        self.label = label
        self.type = type
        self.num_deltas = num_deltas
        self.deltas = deltas
        self.num_cores = num_cores
        self.cores = cores
        self.minutiaes = minutiaes


def get_end_points(point, angle, length):
    y, x = point

    y_final = int(y + length * math.sin(angle))
    x_final = int(x + length * math.cos(angle))

    return y_final, x_final

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

def enhance_image( img):
    a = 150
    y = 95

    enhanced = img

    mean = np.mean(img)
    deviation = np.std(img)
    #percorre a imagem, normalizando-a
    for j in range(len(img)):
        for k in range(len(img[j])):
            enhance = a + y * ((img[j,k] - mean) / deviation)

            if enhance > 255:
                enhanced[j,k] = 255
            elif enhance < 0:
                enhanced[j,k] = 0
            else:
                enhanced[j,k] = enhance

    img = enhanced
    #cv2.imshow("Adding fingerprints to traning set...", enhanced)
    #cv2.waitKey(50)
    # cv2.imwrite("original.png", images[1])

    return mean, deviation

def orientation( img):
    w = 11
    w2d = 1 / math.pow(w, 2)
    alpha_x = []
    alpha_y = []

    iMedian = cv2.medianBlur(img, 5)

    testeimg = np.zeros((300,300), dtype=np.uint8)
    testeimgc = cv2.cvtColor(testeimg, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(iMedian, cv2.COLOR_GRAY2BGR)
    #cv2.imshow("Adding fingerprints to traning set...", iMedian)
    #cv2.waitKey(50)
    sobelx = cv2.Sobel(iMedian,cv2.CV_64F,dx=1,dy=0,ksize=3)
    sobely = cv2.Sobel(iMedian,cv2.CV_64F,dx=0,dy=1,ksize=3)

    alphax = np.subtract(sobelx*sobelx, sobely*sobely) #resulta em matriz 300x300
    alphay = np.multiply((2 * sobelx), sobely) #resulta em matriz 300x300

    for i in range(0,300,w):
        for j in range(0,300,w):
            alpha_x.append(alphax[i:i+w, j:j+w].sum() * w2d) #aplica kernel em toda a matriz alphax (divide por w²)
            alpha_y.append(alphay[i:i+w, j:j+w].sum() * w2d) #aplica kernel em toda a matriz alphay (divide por w²)

    alpha_x = np.reshape(alpha_x, (int(300/w+1), int(300/w+1)))
    alpha_y = np.reshape(alpha_y, (int(300/w+1), int(300/w+1)))
    kernel = np.array(([1,1,1],[1,2,1],[1,1,1]))
    a = cv2.filter2D(alpha_x, -1, kernel)
    b = cv2.filter2D(alpha_y, -1, kernel)

    o_cortado = ((np.arctan2(b, a)/2) + math.pi/2).flatten()

    #o_cortado[abs(o_cortado) > (math.pi/2)] -= math.pi
    k=0
    # for i in range(0,300,w):
    #     for j in range(0,300,w):
    #         y_final, x_final  = get_end_points((i,j), o_cortado[k], 6)
    #         k+=1

            # cv2.line(img, (j,i), (x_final,y_final), (0,0,255), 1)
            # cv2.line(testeimgc, (j,i), (x_final,y_final), (0,0,255), 1)
            #
            # cv2.imshow('norm', img)
            # cv2.waitKey()
            # cv2.imshow('norm', testeimgc)
            # cv2.waitKey()
            # cv2.imshow('norm', img)
            # cv2.waitKey()

    return o_cortado

def region_of_interest( img):
    w0 = w1 = 0.5
    w = 11
    center = 150
    #distancia euclidiana entre (0,0) e (150,150)
    distance = math.sqrt(math.pow(0 - center,2) + math.pow(0 - center,2))

    mean = []
    deviation = []
    for k in range(0,300,w):
        for j in range(0,300,w):
            submatrix = img[k:k+w,j:j+w]
            mean.append(np.mean(submatrix))
            deviation.append(np.std(submatrix))

    min_mean = min(mean)
    max_mean = max(mean)
    min_deviation = min(deviation)
    max_deviation = max(deviation)


    for k in range(0,300-w+1,w):
        for j in range(0,300-w+1,w):

            submatrix = img[k:k+w,j:j+w]
            mean = np.mean(submatrix)
            mean_normalized = (mean-min_mean)/(max_mean-min_mean)
            deviation = np.std(submatrix)
            deviation_normalized = (deviation-min_deviation)/(max_deviation-min_deviation)
            #print k, j, img[k,j]
            w2 = math.sqrt(math.pow(k+(w/2) - center,2) + math.pow(j+(w/2) - center,2))
            w2_normalized = w2 / distance
            w2 = 1 - w2_normalized

            value = w0 * (1 - mean_normalized) + w1 * deviation_normalized + w2
            if (value < 0.8): #o bloco não presta
                img[k:k+w,j:j+w] = 255

    # for i in range(len(images)):
    #     cv2.imwrite(str(i)+".png", images[i])
    #     cv2.waitKey()

def image_binarization( img):
    w = 11
    x = 8

    flattened = img.flatten()
    index_in_order = np.argsort(flattened)
    p25 = flattened[index_in_order[int((len(flattened)/100)*25)]]
    p50 = flattened[index_in_order[int((len(flattened)/100)*50)]]

    for k in range(0,300):
        for j in range(0,300):
            s = img[k,j]
            if s < p25:
                img[k,j] = 0
            elif s > p50:
                img[k,j] = 255
            else:
                aux = img[k:k+x, j:j+x].sum() / x
                submatrix = img[k:k+w,j:j+w]
                mean = np.mean(submatrix)
                if aux >= mean:
                    img[k,j] = 255
                else:
                    img[k,j] = 0


    for k in range(0,300):
        for j in range(0,300):
            nw = 0
            nb = 0
            #SMOOTHING USING FILTER 5
            block = img[k:k+5,j:j+5]
            block = block.flatten()
            for i in range(0, len(block)):
                if block[i] == 255:
                    nw = nw + 1
                else:
                    nb = nb + 1

            if nw >= 18:
                img[k,j] = 255
            elif nb >= 18:
                img[k,j] = 0

            nw = 0
            nb = 0

            #SMOOTHING USING FILTER 3
            anotherblock = img[k:k+3,j:j+3].flatten()
            anotherblock = anotherblock.flatten()
            for i in range(0, len(anotherblock)):
                if anotherblock[i] == 255:
                    nw = nw + 1
                else:
                    nb = nb + 1

            if nw >= 5:
                img[k,j] = 255
            elif nb >= 5:
                img[k,j] = 0

    # for i in range(len(images)):
    #     cv2.imwrite(str(i)+".png", images[i])
    #     cv2.waitKey()

def thinning( img):

    imgcopy = img.copy()
    ret, image = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

    image[image == 255] = 1

    aux = skeletonize(image)

    imgcopy[aux == 1] = 0
    imgcopy[aux == 0] = 255

    return imgcopy

    # for i in range(len(skeleton)):
    #     cv2.imwrite(str(i)+"s.png", self.skeleton[i].astype('uint8'))
    #     cv2.waitKey()

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

def difference(diff):
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
                poincare_index[i,j]  = difference(o_cortado[i-1,j-1] - o_cortado[i-1,j])
                poincare_index[i,j] += difference(o_cortado[i-1,j] - o_cortado[i-1,j+1])
                poincare_index[i,j] += difference(o_cortado[i-1,j+1] - o_cortado[i,j+1])
                poincare_index[i,j] += difference(o_cortado[i,j+1] - o_cortado[i+1,j+1])
                poincare_index[i,j] += difference(o_cortado[i+1,j+1] - o_cortado[i+1,j])
                poincare_index[i,j] += difference(o_cortado[i+1,j] - o_cortado[i+1,j-1])
                poincare_index[i,j] += difference(o_cortado[i+1,j-1] - o_cortado[i,j-1])
                poincare_index[i,j] += difference(o_cortado[i,j-1] - o_cortado[i-1,j-1])

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

def matching(label, imagem_original, imagens_teste):
    w = 11
    M = 10 #10 minúcias
    R = 10 #10 pixels

    imagem = imagem_original.copy()
    print( "Preparando imagem para o matching")
    enhance_image(imagem)
    o_cortado_imagem = orientation(imagem)
    region_of_interest(imagem)
    tipo_imagem, deltas_imagem, cores_imagem = singular_point(o_cortado_imagem, imagem)
    image_binarization(imagem) #DEMORADO
    minucias_imagem = get_minutiae(thinning(imagem))
    minucias_imagem = remove_spurious(minucias_imagem)
    b = []
    if (len(cores_imagem)):
        b.append(np.amax(cores_imagem, axis=0))
    if (len(deltas_imagem)):
        b.append(np.amax(deltas_imagem, axis=0))
    if (len(b)):
        origem_imagem = np.amax(b, axis=0)
        translate(minucias_imagem, origem_imagem)
        angle_imagem = o_cortado_imagem[origem_imagem[0]*origem_imagem[1]]
        rotate(minucias_imagem, origem_imagem, angle_imagem)

    maior = 0
    for teste in imagens_teste:
        if (tipo_imagem == teste.type):
            print("Calculando matching de {} com {}: ").format(label, teste.label),

            # calcular score das minucias
            s = (100/M)*sum_minutiaes(M, R, teste.minutiaes, minucias_imagem)
            print( s)
            if (s > maior):
                maior = s
                label_match = teste.label

    print ("O melhor match de {} é a imagem {}").format(label, label_match),
    if (label == label_match):
        print( ", e felizmente, são as mesmas imagens.")
    else:
        print( ", porém, são imagens diferentes.")

def sum_minutiaes(M, max_raio, minucias_teste, minucias):
    soma = 0
    for i in range(1, M):
        r = math.sqrt(math.pow(minucias_teste[i][0] - minucias[i][0],2) + math.pow(minucias_teste[i][1] - minucias[i][1],2))
        soma += (1-r)/max_raio #r é a distancia de duas minúcias

    return soma

def translate(minutiaes, origem):
    w = 11
    o = origem.copy()
    #posição de cada minúcia menos a posição da origem (singularidade mais alta)
    o[0] = o[0]*w
    o[1] = o[1]*w
    for m in minutiaes:
        m[0] = m[0]-o[0]
        m[1] = m[1]-o[1]

    return

def rotate(minutiaes, origem, angle):
    mat = cv2.getRotationMatrix2D((int(origem[0]),int(origem[1])), angle, 1)
    minutiaes_array = np.asarray(minutiaes)
    minutiaes = cv2.warpAffine(minutiaes_array, mat, (300,300), flags=cv2.INTER_NEAREST)

    return

np.set_printoptions(threshold=sys.maxsize)
dataset_path = './data/Lindex101'
lindex = Dataset()
print("Lendo dataset Lindex101")
lindex.get_images_and_labels(dataset_path)
lindex.paths.sort()
i=0

#Criar o vetor de vetores para as imagens usadas no matching
for img_o in lindex.images:
    img = img_o.copy()
    print("Imagem "+str(i+1))
    enhance_image(img)
    print("Calculando orientação")
    o_cortado = orientation(img)
    region_of_interest(img)
    print( "Encontrando pontos singulares")
    tipo, deltas, cores = singular_point(o_cortado, img)
    print( "Binarizando a imagem. Isso pode demorar alguns segundos.")
    image_binarization(img) #DEMORADO
    lindex.skeleton.append(thinning(img))
    print( "Encontrando minúcias e removendo spurious")
    minutiaes = get_minutiae(lindex.skeleton[i])
    minutiaes = remove_spurious(minutiaes) #vetor de posições
    #cv2.imwrite(str(i)+".png", img)
    #cv2.waitKey()
    if minutiaes: #imagem possui minúcias
         image = lindex.skeleton[i].copy()
         for (x,y) in minutiaes:
             image[y-1:y+2,x-1:x+2] = 0 #marca a minúcia
         cv2.imwrite(str(i)+".png", image)
         cv2.waitKey(200)

    #pegar a singularidade mais alta, ou core ou delta
    biggest = []
    if (len(cores)):
        biggest.append(np.amax(cores, axis=0))
    if (len(deltas)):
        biggest.append(np.amax(deltas, axis=0))
    if (len(biggest)):
        big_biggest = np.amax(biggest, axis=0)
        biggest = []
        origem = big_biggest
        print( "Transladando minúcias")
        translate(minutiaes, origem)
        angle = o_cortado[origem[0]*origem[1]]
        print( "Rotacionando minúcias")
        rotate(minutiaes, origem, angle)
    imagem_teste = Teste(lindex.paths[i], tipo, len(deltas), deltas, len(cores), cores, minutiaes)
    print ("Tipo: "+imagem_teste.type)
    print ("\n")
    lindex.testes.append(imagem_teste)
    i+=1

i = 0
for img in lindex.images:
    matching(lindex.paths[i], img, lindex.testes)
    i+=1
