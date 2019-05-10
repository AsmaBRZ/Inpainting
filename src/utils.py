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

###################################
#                                 #
#           PART I                #
#                                 #
###################################

#Load USPS data
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

#Display USPS data
def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

#we filter the data so as to keep only the specific number in the arguments VS all the orther numbers
def filter_oneVSall(x, y, number):
    datay = np.copy(y)
    datay[y == number] = -1
    datay[y !=number ] = 1
    return x,datay

#we filter the data so as to keep only the two specific numbers in the arguments
def filter_oneVsone(x, y, number1,number2):
    datax, datay = np.copy(x), np.copy(y)
    i = np.where(np.logical_or(datay == number1, datay == number2))[0]
    new_datay= np.array(datay[i])
    new_datay[new_datay == number1] = -1
    new_datay[new_datay == number2] = 1
    return np.array(datax[i]),new_datay

#we calculate the score obtained after the prediction
def score(datay, datay_predicted):
    return np.mean((np.sign(datay_predicted) == datay[:, np.newaxis]))



###################################
#                                 #
#            PART II              #
#                                 #
###################################

#A usefull function to normalize an image to display
def normalize(X):
    Xmin = float(X.min())
    Xmax = float(X.max())
    return (X - Xmin)/(Xmax - Xmin)

#Read an image, convert it to rgb, hsv and hsv normalized
def read_im(fn):
    im_rgb=plt.imread(fn)
    im_hsv=colors.rgb_to_hsv(im_rgb[:,:,:3])
    im_hsv_norm=normalize(im_hsv)
    return im_rgb,im_hsv,im_hsv_norm

#Display an image
def show_im(im):
    plt.imshow(im)

#Return the patch centered in (i, j)
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

#convert patch to vector
def patchToVector(patch):
    return patch.flatten()
#convert vector to path. I specified N and M because it's not all imges that are squared.
def vectorToPatch(vect,N,M):
    return np.reshape(vect, (N, M, 3))

#generate noie on a random pixels with a fixed percentage
def noise(img,prc):
    if prc == 0.0: return img
    img_noisy=np.copy(img)
    N=img.shape[0]
    M=img.shape[1]
    nb_pix=N*M
    nb_pix_noisy=int((prc*nb_pix)/100)
    for pix in range(nb_pix_noisy):
        x=np.random.choice(np.arange(0,N,1))
        y=np.random.choice(np.arange(0,M,1))
        img_noisy[x][y]=[253,253,253]
    return img_noisy

#remove a whole rectangle from the image at the position i,j
def delete_rect(img,i,j,height,width):
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
            img_noisy[w][x]=[253,253,253]
    return img_noisy

#From an image, we extract its patches
def get_patches(im,step,h):
    patches=[]
    N,M,_=im.shape
    axis_n=np.arange(0,N+1,step)
    axis_m=np.arange(0,M+1,step)
    #devide the image into patches
    for i in range(1,len(axis_n)-1):
        for j in range(1,len(axis_m)-1):
            patches.append(get_patch(axis_n[i],axis_m[j],h,im))
    return np.array(patches)

#From all the patches of the image, we only extract the incomplete patches
def getPatches_incomp(patches):
    patches_incomp=[]
    for patch in patches:
            if 253 in patchToVector(patch):
                patches_incomp.append(patch)
    return np.array(patches_incomp)

#From all the patches of the image, we only extract the complete patches
def getPatches_comp(patches):
    patches_comp=[]
    for patch in patches:
            if 253 not in patchToVector(patch):
                patches_comp.append(patch)
    return np.array(patches_comp)

#This function determines if a patch if complete or it contains a missing pixels
def isComp(patch,patches_comp):
    for p in patches_comp:
        if np.array_equal(p.flatten(),patch.flatten()):
            return True
    return False

#We build from our set of completes patches the data for the train step
def build_data_train(im,patches_comp,step,h):
    #data_train contains two vectors, the first vector is datax and the second one is datay (the pixel to predict)
    data_train=[[],[]]
    N,M,_=im.shape
    axis_n=np.arange(0,N+1,step)
    axis_m=np.arange(0,M+1,step)
    #devide the image into patches
    for i in range(1,len(axis_n)-1):
        for j in range(1,len(axis_m)-1):
            patch=get_patch(axis_n[i],axis_m[j],h,im)
            #if the patch if complete, we add it to the data set
            if isComp(patch,patches_comp):
                #we delete the pixel at the center, the one to predict
                p=np.delete(patchToVector(patch),len(patch)/2)
                p=np.delete(p,len(p)/2)
                p=np.delete(p,len(p)/2)
                data_train[0].append(p)
                data_train[1].append(im[axis_n[i]][axis_m[j]])
    return data_train

#We use a model (lasso in our case) to predict the value of missing mixels of an image
def reconstruct_im(model,img, h):
    result=img.copy()
    for i in range(h//2, len(result) - h//2):
        for j in range(h//2, len(result[i]) - h//2):
            if result[i][j][0]==253 and result[i][j][1]==253 and result[i][j][2]==253:
                patch=patchToVector(get_patch(i, j, h,result))
                index = np.argwhere(np.equal(patch, 253))[0:3]
                patch = np.delete(patch, index)
                result[i][j] = model.predict([patch])[0]
    return result

#From an image im, a pixel at i,j, this function calculate the frame surronding the pixel at i,j with height=wight=h
def get_Frame(i, j, h, im):

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
    return a,b,c,d

def reconstruct_im_rec(model,im, h,p,q):
    img=im.copy()
    patch=get_patch(p, q, h,im)
    #The size of the patch we'll be used later in prediction. We delete 3 pixels to consider the pixel missing
    size_p=len(patch.flatten())-3
    #we calculate the 4 coins defining the patch on the original image
    c0,c1,c2,c3=get_Frame(p,q, h,img)
    patch_im=patch.copy()
    #la and lb are the limits of the variables a and b
    a=c0
    la=c1
    b=c2
    lb=c3
    #We begin from the frame of the patch. We predict all its values. Then, at each iteration we consider the frame surrounded by the previous frame
    while a!=la and b!=lb :
        #We predict the two verticals and then the ewo horizantals bars of the frame
        if(b<= lb):
            for bb in range(b,lb+1):
                #at the missed pixel, we consider a patch as a tmp_patch
                patch_tmp = get_patch(a, bb, h,img)
                patch_tmp=patch_tmp.flatten()
                #we delete all values containing the noise
                index_tmp=np.argwhere(np.equal(patch_tmp,253))
                patch_tmp = np.delete(patch_tmp, index_tmp)
                #we calculate the median of the values of the tmp_patch
                med=int(np.median(patch_tmp))
                #Depending on the position of the pixel to predict, we may delete a various number of a noisy pixels
                #The problem is that the tmp_patch to predict now doest have the same size as the initial patch
                #as a solution, we fill with the median value until we attend the patch's size
                while len(patch_tmp)<size_p:
                    patch_tmp=np.append(patch_tmp,med)
                img[a][bb] = model.predict([patch_tmp])[0]
                patch_tmp = get_patch(la, bb, h,img)
                patch_tmp=patch_tmp.flatten()
                index_tmp=np.argwhere(np.equal(patch_tmp, 253))
                patch_tmp = np.delete(patch_tmp, index_tmp)
                med=int(np.median(patch_tmp))
                while len(patch_tmp)<size_p:
                    patch_tmp=np.append(patch_tmp,med)
                img[la][bb] = model.predict([patch_tmp])[0]
        if(a<= la):
            for aa in range(a,la+1):
                #we detect a missed pixel
                patch_tmp = get_patch(aa, b, h,img)
                patch_tmp=patch_tmp.flatten()
                index_tmp=np.argwhere(np.equal(patch_tmp, 253))
                patch_tmp = np.delete(patch_tmp, index_tmp)
                med=int(np.median(patch_tmp))
                while len(patch_tmp)<size_p:
                    patch_tmp=np.append(patch_tmp,med)
                img[aa][b] = model.predict([patch_tmp])[0]
                #we detect a missed pixel
                patch_tmp = get_patch(aa, lb, h,img)
                patch_tmp=patch_tmp.flatten()
                index_tmp=np.argwhere(np.equal(patch_tmp, 253))
                patch_tmp = np.delete(patch_tmp, index_tmp)
                med=int(np.median(patch_tmp))
                while len(patch_tmp)<size_p:
                    patch_tmp=np.append(patch_tmp,med)
                img[aa][lb] = model.predict([patch_tmp])[0]
        if(a<la):
            a+=1
            la-=1
        if(b<lb):
            b+=1
            lb-=1
    #Now, as it remains the pixel in the center, we predict its value at the end
    patch_tmp = get_patch(p, q, h,img)
    patch_tmp=patch_tmp.flatten()
    index_tmp=np.argwhere(np.equal(patch_tmp, 253))
    patch_tmp = np.delete(patch_tmp, index_tmp)
    med=int(np.median(patch_tmp))
    while len(patch_tmp)<size_p:
        patch_tmp=np.append(patch_tmp,med)
    img[p][q] = model.predict([patch_tmp])[0]
    return img