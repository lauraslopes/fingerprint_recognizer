import numpy as np
import cv2
import math
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

def pre_process(img):
    enhancement(img)
    print("Calculando orientação")
    o_cortado = orientation(img)
    region_of_interest(img)
    binarization(img) #DEMORADO
    smoothing(img)

    return img, o_cortado