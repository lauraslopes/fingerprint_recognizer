import numpy as np
import cv2
import math

from fingerprint import Fingerprint, singular_point, remove_spurious, get_minutiae
from image import enhancement, binarization, smoothing, thinning, region_of_interest, orientation

def matching(label, imagem_original, imagens_teste):
    w = 11
    M = 10 #10 minúcias
    R = 10 #10 pixels

    imagem = imagem_original.copy()
    print( "Preparando imagem para o matching")
    enhancement(imagem)
    o_cortado_imagem = orientation(imagem)
    region_of_interest(imagem)
    tipo_imagem, deltas_imagem, cores_imagem = singular_point(o_cortado_imagem, imagem)
    binarization(imagem) #DEMORADO
    smoothing(imagem)
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
            print("Calculando matching de {} com {}: ".format(label, teste.label))

            # calcular score das minucias
            s = (100/M)*sum_minutiaes(M, R, teste.minutiaes, minucias_imagem)
            print( s)
            if (s > maior):
                maior = s
                label_match = teste.label

    label_match = ""

    print ("O melhor match de {} é a imagem {}".format(label, label_match))
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