#!/usr/bin/python
# -*- coding: latin-1 -*-


# Import the required modules
import cv2, os
import sys
import numpy as np
 
sys.path.insert(0, os.path.abspath('./src'))
from dataset import Dataset
from fingerprint import Fingerprint, singular_point, remove_spurious, get_minutiae
from image import pre_process, thinning
from match_fingerprints import matching, translate, rotate

np.set_printoptions(threshold=sys.maxsize)
lindex = Dataset(sys.argv[1])
lindex.get_images_and_labels()
lindex.paths.sort()
i=0

#Criar o vetor de vetores para as imagens usadas no matching
for img_o in lindex.images:
    img, o_cortado = pre_process(img_o.copy())

    print( "Encontrando pontos singulares")
    tipo, deltas, cores = singular_point(o_cortado, img)

    skeleton = thinning(img)
    print( "Encontrando minúcias e removendo spurious")
    minutiaes = get_minutiae(skeleton)
    minutiaes = remove_spurious(minutiaes) #vetor de posições

    #cv2.imwrite(str(i)+".png", img)
    #cv2.waitKey()
    '''
    if minutiaes: #imagem possui minúcias
         image = skeleton.copy()
         for (x,y) in minutiaes:
             image[y-1:y+2,x-1:x+2] = 0 #marca a minúcia
         cv2.imwrite(str(i)+".png", image)
         cv2.waitKey(200)
    '''
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
    imagem_teste = Fingerprint(lindex.paths[i], tipo, len(deltas), deltas, len(cores), cores, minutiaes)
    print ("Tipo: "+imagem_teste.type)
    print ("\n")
    lindex.testes.append(imagem_teste)
    i+=1

i = 0
for img in lindex.images:
    matching(lindex.paths[i], img, lindex.testes)
    i+=1
