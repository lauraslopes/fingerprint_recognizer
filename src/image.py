import numpy as np
import cv2
from skimage.morphology import skeletonize

def enhancement(img):
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

    return mean, deviation


def binarization(img):
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

    return img

def smoothing(img):
    for k in range(0,300):
        for j in range(0,300):
            nw = 0
            nb = 0
            #USING FILTER 5
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

            #USING FILTER 3
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

    return img


def thinning(img):

    imgcopy = img.copy()
    _,image = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

    image[image == 255] = 1

    aux = skeletonize(image)

    imgcopy[aux == 1] = 0
    imgcopy[aux == 0] = 255

    return imgcopy