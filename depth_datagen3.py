# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:13:12 2020

@author: boonping
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage.interpolation import rotate
from skimage.draw import line_aa
from skimage.draw import polygon

siz=2
samp=np.array([])
result=np.array([])


#straightwall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    centery=np.random.randint(256,512)
    centerx=np.random.randint(385,1407)
    
    centery2=np.random.randint(0,256)
    centerx2=np.random.randint(385,1407)
    
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    wallpointx=256*np.ones(wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    
    poly2=np.append(wallpointx,centerx2)
    polyy2=np.append(wallpoint,centery2)
    
    z=np.random.randint(67,252)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    rr,cc=polygon(poly2,polyy2)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for w in wallpoint:
        
        print("218",w)
        rr,cc,val=line_aa(256,w,centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255) #z*(rr-centerx)/
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255) #z*(rr-centerx)/
        rr,cc,val=line_aa(256,w,centerx2,centery2)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
        #print(rr,cc,np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255))
        #print(arr[rr,cc],arr2[rr,cc])

        #print(np.nonzero(arr[:,129]))
        #print(np.nonzero(arr2[:,129]))

    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)

    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    if 0:
        if 1:
            hh,hhy=np.nonzero(arr)
            
            print(hh,hhy,arr[hh,hhy],arr2[hh,hhy])
            print("282a",hh[0],hhy[0],arr[hh[0],hhy[0]],arr2[hh[0],hhy[0]],np.clip( (z*(hh[0]-centerx)/(256-centerx))+3, 64, 255))

            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])



#straightwall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    wallpointx=256*np.ones(wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(67,252)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for w in wallpoint:
        
        print(w)
        rr,cc,val=line_aa(256,w,centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0        
    arr[arr>255]=255
    arr2[arr>255]=255
        
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)

    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255

    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#straightwall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    centery=np.random.randint(0,256)
    centerx=np.random.randint(385,1407)
    centery2=np.random.randint(256,512)
    centerx2=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    wallpointx=256*np.ones(wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    poly2=np.append(wallpointx,centerx2)
    polyy2=np.append(wallpoint,centery2)
    z=np.random.randint(67,252)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    rr,cc=polygon(poly2,polyy2)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for w in wallpoint:
        
        print("213",w)
        rr,cc,val=line_aa(256,w,centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        rr,cc,val=line_aa(256,w,centerx2,centery2)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
        
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)

    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255

    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
   
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#uneven wall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(67,251)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for ii in range(wallpoint.shape[0]):
        
        print("214",wallpoint[ii])
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)

    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
#raise    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#uneven wall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    
    centery=np.random.randint(0,256)
    centerx=np.random.randint(385,1407)
    centery2=np.random.randint(256,512)
    centerx2=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    poly2=np.append(wallpointx,centerx2)
    polyy2=np.append(wallpoint,centery2)
    z=np.random.randint(67,251)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    rr,cc=polygon(poly2,polyy2)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for ii in range(wallpoint.shape[0]):
        
        print("215",wallpoint[ii])
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx2,centery2)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        

    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#uneven wall
for kk in range(siz):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(32,64)
    
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(67,251)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for ii in range(wallpoint.shape[0]):
        
        print("219",wallpoint[ii])
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    for jj in range(3):
        ran=np.random.randint(0,255)
        rany=np.random.randint(0,255)
        #print(ran)
        q=np.random.randint(64,255)
        arr[ran,rany]=q
        arr2[ran,rany]=q
        
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
        
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])



#uneven wall
for kk in range(siz*2):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(2,31)
    
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(67,252)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for ii in range(wallpoint.shape[0]):
        
        print("220",wallpoint[ii])
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    for jj in range(3):
        ran=np.random.randint(0,255)
        rany=np.random.randint(0,255)
        #print(ran)
        q=np.random.randint(64,255)
        arr[ran,rany]=q
        arr2[ran,rany]=q
        
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
        
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

#uneven wall
for kk in range(siz*2):
    arr=np.zeros((1408,1408))
    arr2=np.zeros((1408,1408))
    pitch=np.random.randint(2,31)
    
    centery=np.random.randint(0,256)
    centerx=np.random.randint(385,1407)
    centery2=np.random.randint(256,512)
    centerx2=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    poly2=np.append(wallpointx,centerx2)
    polyy2=np.append(wallpoint,centery2)
    z=np.random.randint(67,252)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    rr,cc=polygon(poly2,polyy2)
    arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
    
    for ii in range(wallpoint.shape[0]):
        
        print("221",wallpoint[ii])
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx2,centery2)
        arr[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        arr2[rr,cc]=np.clip( (z*(rr-centerx)/(256-centerx))+3, 64, 255)
        
    arr=arr.astype('int')
    arr2=arr2.astype('int')
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    for jj in range(3):
        ran=np.random.randint(0,255)
        rany=np.random.randint(0,255)
        #print(ran)
        q=np.random.randint(64,255)
        arr[ran,rany]=q
        arr2[ran,rany]=q
        
    arr[arr<64]=0
    arr2[arr2<64]=0
    arr[arr>255]=255
    arr2[arr>255]=255
        
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

'''
#uneven wall
for kk in range(siz*2):
    lev=np.random.randint(85,223)
    arr=np.random.randint(lev-20,lev+20,size=(1408,1408))
    arr[arr<64]=0
    #arr2=np.copy(arr)
    arr2=lev*np.ones((1408,1408))
    pitch=np.random.randint(2,31)
    
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(245,255)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=((253-z)*(rr-centerx)/(256-centerx))+z
    
    for ii in range(wallpoint.shape[0]):
        
        #print(w)
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=((253-z)*(rr-centerx)/(256-centerx))+z #z*(rr-centerx)/
        
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    for jj in range(3):
        ran=np.random.randint(0,255)
        rany=np.random.randint(0,255)
        #print(ran)
        q=np.random.randint(64,255)
        arr[ran,rany]=q
        arr2[ran,rany]=q
        
        
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise

   
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#uneven wall
for kk in range(siz):
    lev=np.random.randint(85,223)
    arr=np.random.randint(lev-20,lev+20,size=(1408,1408))
    arr[arr<64]=0
    #arr2=np.copy(arr)
    arr2=lev*np.ones((1408,1408))
    pitch=np.random.randint(32,63)
    
    centery=np.random.randint(0,512)
    centerx=np.random.randint(385,1407)
    starty=np.random.randint(0,8)
    wallpoint= np.arange(512+starty,767,pitch) 
    #wallpointx=256*np.ones()
    wallpointx=256+np.random.randint(-10,10,size=wallpoint.shape)
    poly=np.append(wallpointx,centerx)
    polyy=np.append(wallpoint,centery)
    z=np.random.randint(245,255)
    print(poly,polyy)
    rr,cc=polygon(poly,polyy)
    arr2[rr,cc]=((253-z)*(rr-centerx)/(256-centerx))+z
    
    for ii in range(wallpoint.shape[0]):
        
        #print(w)
        rr,cc,val=line_aa(wallpointx[ii],wallpoint[ii],centerx,centery)
        arr[rr,cc]=((253-z)*(rr-centerx)/(256-centerx))+z #z*(rr-centerx)/
        
    
    shi=np.random.randint(-50,50)
    shi2=np.random.randint(-50,50)
    arr=np.roll(arr,shi,axis=0)
    arr=np.roll(arr,shi2,axis=1)
    arr2=np.roll(arr2,shi,axis=0)
    arr2=np.roll(arr2,shi2,axis=1)
    
    
    arr=arr[128:384,512:768]
    arr2=arr2[128:384,512:768]
    
    
    ang=np.random.randint(-179,180)
    arr=rotate(arr, angle=ang, reshape=False)
    arr2=rotate(arr2, angle=ang, reshape=False)
    
    for jj in range(3):
        ran=np.random.randint(0,255)
        rany=np.random.randint(0,255)
        #print(ran)
        q=np.random.randint(64,255)
        arr[ran,rany]=q
        arr2[ran,rany]=q
        
        
    if 0:
        if 1:
            x=np.arange(0,256)
            y=np.arange(0,256)
            
            ax1 = plt.subplot(211)
            ax1.set_aspect('equal')
            
            # equivalent but more general
            ax1.pcolormesh(x,y,arr)

            ax2 = plt.subplot(212)
            ax2.set_aspect('equal')
            
            # equivalent but more general
            ax2.pcolormesh(x,y,arr2)
             
            
            plt.show()
            raise
        
    samp=np.append(samp,arr.astype('int'))
    result=np.append(result,arr2.astype('int'))
    a=arr
    b=arr2
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.flipud(a)
    b=np.flipud(b)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=np.fliplr(a)
    b=np.fliplr(b)
    #print(a.shape)
    
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
    #raise

   
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])
'''

'''
with open("input.csv", "r") as f:
    samp=np.loadtxt(f)
with open("output.csv", "r") as f:
    result=np.loadtxt(f)
samp=samp.reshape(int(samp.shape[0]/256/256),256,256)
result=result.reshape(int(result.shape[0]/256/256),256,256)

print(samp.shape,result.shape)
samp=np.array([])
result=np.array([])
'''