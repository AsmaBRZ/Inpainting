import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linalg as LA
import copy 
import sklearn.feature_extraction.image as sk_fe
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.linear_model import (LinearRegression,Ridge, RidgeClassifierCV,
                                  LassoCV, Lasso,Ridge)
from sklearn.manifold import Isomap
import matplotlib.colors as colors


def normalize(X):
    Xmin = float(X.min())
    Xmax = float(X.max())
    return (X - Xmin)/(Xmax - Xmin)

def read_im(fn):
    im_rgb=plt.imread(fn)
    im_hsv=colors.rgb_to_hsv(im_rgb[:,:,:3])
    im_hsv_norm=normalize(im_hsv)
    return im_rgb,im_hsv,im_hsv_norm

def show_im(im):
    plt.imshow(im)

def get_patch(i,j,h,im):
    N,M,_=im.shape
    a=0
    b=0
    c=0
    d=0
    if h%2==0:
        #pair
        h_p=int(h/2)
        if i-h_p>=0:
            a=i-h_p
        else:
            a=0
        if i+h_p<=N:
            b=i+h_p
        else:
            b=N
        if j-h_p>=0:
            c=j-h_p
        else:
            c=0
        if j+h_p<=M:
            d=j+h_p
        else:
            d=M
    else:
        #impair
        h_i=int(round(h/2))
        if i-h_i>=0:
            a=i-h_i
        else:
            a=0
        if i+h_i+1<=N:
            b=i+h_i+1
        else:
            b=N
        if j-h_i>=0:
            c=j-h_i
        else:
            c=0
        if j+h_i+1<=M:
            d=j+h_i+1
        else:
            d=M
    return im[a:b,c:d]

def getNoiseValue(image):
    v=patchToVector(image)
    union=np.arange(0.0,256.0,0.05)
    return np.random.choice(np.setdiff1d(union,v))
    
def patchToVector(patch):
    p=patch
    return p.flatten()

def vectorToPatch(vect,a,b):
    return np.reshape(vect, (a, b, 3))

def noise(img,prc,value):
    if prc == 0.0: return img
    img_noisy=np.copy(img)
    N=img.shape[0]
    M=img.shape[1]
    nb_pix=N*M
    nb_pix_noisy=int((prc*nb_pix)/100)
    
    for pix in range(nb_pix_noisy):
        x=np.random.choice(np.arange(0,N,1))
        y=np.random.choice(np.arange(0,M,1))
        img_noisy[x][y][0]=value[0]
        img_noisy[x][y][1]=value[1]
        img_noisy[x][y][2]=value[2]
    return img_noisy

def delete_rect(img,i,j,height,width,value):
    img_noisy=np.copy(img)
    N,M,_=img.shape
    a=0
    b=N
    c=0
    d=M
    kk=0
    if width % 2==0:
        w_p=int(width/2)
        if i-w_p>=0:
            a=i-w_p
        else:
            a=0
        if i+w_p<=N:
            b=i+w_p
            kk=1
        else:
            b=N
    else:
        w_i=int(width/2)
        if i-w_i>=0:
            a=i-w_i
        else:
            a=0
        if i+w_i+1<=N:
            b=i+w_i+1
            kk=1
        else:
            b=N
            
    if height % 2==0:
        h_p=int(height/2)
        if j-h_p>=0:
            c=j-h_p
        else:
            c=0
        if j+h_p<=M:
            d=j+h_p
        else:
            d=M
    else:
        h_i=int(height/2)
        if j-h_i>=0:
            c=j-h_i
        else:
            c=0
        if j+h_i+1<=M:
            d=j+h_i+1
        else:
            d=M
    for w in range(a,b):
        for x in range(c,d):
            img_noisy[w][x][0]=value[0]
            img_noisy[w][x][1]=value[1]
            img_noisy[w][x][2]=value[2]
    return img_noisy

def build_dictionary(im,step,h,value):
    N,M,_=im.shape
    axis_n=np.arange(0,N+1,step)
    axis_m=np.arange(0,M+1,step)
    patches_incomp=[]
    patches_comp=[]
    #devide the image into patches
    for i in range(1,len(axis_n)-1):
        for j in range(1,len(axis_m)-1):
            patch=get_patch(axis_n[i],axis_m[j],h,im)
            if np.any(np.any(patch== np.array([value[0],value[1],value[2]]))):
               #patch detected
                patches_incomp.append(patch)
            else:
                patches_comp.append(patch)

    return patches_incomp,patches_comp

def build_data(patches):
	datax=[]
	datay=[]
	for i,patch in enumerate(patches):
		datax.append(patch)
		datay.append(np.delete(patches,i))	

	return datax, datay