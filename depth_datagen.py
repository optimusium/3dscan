# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:13:12 2020

@author: boonping
"""

import numpy as np
import matplotlib.pyplot as plt
import random

siz=32
samp=np.array([])
result=np.array([])


#cylinder
for i in range(siz):
    r=np.random.randint(6,255)
    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            if rr>r**2: continue
            if random.random()>0.6:
                a[i][j]=z
            b[i][j]=z
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,a.astype('int'))
    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])



#cone
for i in range(siz):
    r=np.random.randint(6,255)
    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,a.astype('int'))
    
    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])
   

#conal disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,a.astype('int'))
    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

#disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=z
            b[i][j]=z
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

#oval disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=z
            b[i][j]=z
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=np.random.randint(8,126)
    a=np.append(np.zeros((ad,256)),a)
    a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#oval 
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=z
            b[i][j]=z
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=np.random.randint(8,126)
    a=np.append(np.zeros((ad,256)),a)
    a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

#oval 
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=z
            b[i][j]=z
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=128
    a=np.append(np.zeros((ad,256)),a)
    #a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    #b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#conal oval 
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=128
    a=np.append(np.zeros((ad,256)),a)
    #a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    #b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#conal oval disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=128
    a=np.append(np.zeros((ad,256)),a)
    #a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    #b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#conal oval disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=np.random.randint(8,128-2)
    a=np.append(np.zeros((ad,256)),a)
    a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#conal oval disc
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    for i in range(256):
        for j in range(256):
            rr=(i-c[0])**2+(j-c[1])**2
            rr=rr**0.5
            if rr>r: continue
            #if rr<r2: continue
            if random.random()>0.6 :
                a[i][j]=int(64+(z-64)*(r-rr)/r)
            b[i][j]=int(64+(z-64)*(r-rr)/r)
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
            
    a2=a[np.arange(0,256,2)]
    a=a2 
    b2=b[np.arange(0,256,2)]
    b=b2 
    ad=np.random.randint(8,128-2)
    a=np.append(np.zeros((ad,256)),a)
    a=np.append(a,np.zeros((128-ad,256)))
    a=a.reshape(int(a.shape[0]/256),256)
    b=np.append(np.zeros((ad,256)),b)
    b=np.append(b,np.zeros((128-ad,256)))
    b=b.reshape(int(b.shape[0]/256),256)
    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
    
samp=np.array([])
result=np.array([])


#rectangular
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    x1=np.random.randint(6,253)
    x2=np.random.randint(x1+2,255)

    y1=np.random.randint(6,253)
    y2=np.random.randint(y1+2,255)
    
    for i in range(256):
        for j in range(256):
            if i>=x1 and i<=x2 and j>=y1 and j<=y2:
                if random.random()>0.6 :
                    a[i][j]=z
                b[i][j]=z
                
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
   
    print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#hollow rectangular
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    x1=np.random.randint(10,243)
    x2=np.random.randint(x1+10,255)

    y1=np.random.randint(10,243)
    y2=np.random.randint(y1+10,255)

    x11=np.random.randint(x1+2,x2-5)
    x12=np.random.randint(x11+2,x2-2)

    y11=np.random.randint(y1+2,y2-5)
    y12=np.random.randint(y11+2,y2-2)
     
    for i in range(256):
        for j in range(256):
            if i>=x11 and i<x12 and j>=y11 and j<y12: continue
            if i>=x1 and i<=x2 and j>=y1 and j<=y2:
                if random.random()>0.6 :
                    a[i][j]=z
                b[i][j]=z

            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
    
    print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))



#triangle
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    p1=np.random.randint(0,20)
    p2=np.random.randint(0,20)
    q1=np.random.randint(20,256*(p1+p2)-21)
    q2=np.random.randint(q1+20,256*(p1+p2))
    
    
    x1=np.random.randint(10,229)
    x2=np.random.randint(x1+10,243)
    x3=np.random.randint(x2+10,255)

    y1=np.random.randint(20,243)
    y2=np.random.randint(y1+10,255)
    y3=np.random.randint(10,y1-9)
    
    poin1=np.append(x1,y1)
    poin2=np.append(x2,y2)
    poin3=np.append(x3,y3)

    polate1=np.append(x1,x2)
    polate1y=np.append(y1,y2)
    polate2=np.append(x2,x3)
    polate2y=np.append(y2,y3)
    polate3=np.append(x1,x3)
    polate3y=np.append(y1,y3)
    #print(poin1,poin2,poin3)
    for i in range(256):
        for j in range(256):
            
            if i<poin1[0] or i>poin3[0]: continue
            #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
            #print(poin1,poin3,np.array([i]),np.interp(np.array([i]),poin1,poin3))
            #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))            
            
            #if i<=poin2[0] and j<=np.interp(np.array([i]),poin1,poin2) and j>=np.interp(np.array([i]),poin1,poin3):
            if i<=poin2[0] and j<=np.interp(i,polate1,polate1y) and j>=np.interp(i,polate3,polate3y):
                #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
                #print(poin1,poin2,i,np.interp(i,poin1,poin2))
                #print(poin1,poin3,i,np.interp(i,poin1,poin3))
                #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))
                if random.random()>0.6 :
                    a[i][j]=z
                b[i][j]=z
            
            #elif i<=poin3[0] and i>poin2[0] and j<=np.interp(np.array([i]),poin2,poin3) and j>=np.interp(np.array([i]),poin1,poin3):
            elif i<=poin3[0] and i>poin2[0] and j<=np.interp(i,polate2,polate2y) and j>=np.interp(i,polate3,polate3y):
                if random.random()>0.6 :
                    a[i][j]=z
                b[i][j]=z
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
                
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))


#tipped triangle
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    p1=np.random.randint(0,20)
    p2=np.random.randint(0,20)
    q1=np.random.randint(20,256*(p1+p2)-21)
    q2=np.random.randint(q1+20,256*(p1+p2))
    
    
    x1=np.random.randint(10,229)
    x2=np.random.randint(x1+10,243)
    x3=np.random.randint(x2+10,255)

    y1=np.random.randint(20,243)
    y2=np.random.randint(y1+10,255)
    y3=np.random.randint(10,y1-9)
    
    poin1=np.append(x1,y1)
    poin2=np.append(x2,y2)
    poin3=np.append(x3,y3)

    polate1=np.append(x1,x2)
    polate1y=np.append(y1,y2)
    polate2=np.append(x2,x3)
    polate2y=np.append(y2,y3)
    polate3=np.append(x1,x3)
    polate3y=np.append(y1,y3)
    #print(poin1,poin2,poin3)
    for i in range(256):
        for j in range(256):
            
            if i<poin1[0] or i>poin3[0]: continue
            #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
            #print(poin1,poin3,np.array([i]),np.interp(np.array([i]),poin1,poin3))
            #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))            
            
            #if i<=poin2[0] and j<=np.interp(np.array([i]),poin1,poin2) and j>=np.interp(np.array([i]),poin1,poin3):
            if i<=poin2[0] and j<=np.interp(i,polate1,polate1y) and j>=np.interp(i,polate3,polate3y):
                #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
                #print(poin1,poin2,i,np.interp(i,poin1,poin2))
                #print(poin1,poin3,i,np.interp(i,poin1,poin3))
                #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(i-poin1[0])
                b[i][j]=64+(z-64)*(i-poin1[0])
            
            #elif i<=poin3[0] and i>poin2[0] and j<=np.interp(np.array([i]),poin2,poin3) and j>=np.interp(np.array([i]),poin1,poin3):
            elif i<=poin3[0] and i>poin2[0] and j<=np.interp(i,polate2,polate2y) and j>=np.interp(i,polate3,polate3y):
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(i-poin1[0])
                b[i][j]=64+(z-64)*(i-poin1[0])
            
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
                
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))



#tipped triangle
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    p1=np.random.randint(0,20)
    p2=np.random.randint(0,20)
    q1=np.random.randint(20,256*(p1+p2)-21)
    q2=np.random.randint(q1+20,256*(p1+p2))
    
    
    x1=np.random.randint(10,229)
    x2=np.random.randint(x1+10,243)
    x3=np.random.randint(x2+10,255)

    y1=np.random.randint(20,243)
    y2=np.random.randint(y1+10,255)
    y3=np.random.randint(10,y1-9)
    
    poin1=np.append(x1,y1)
    poin2=np.append(x2,y2)
    poin3=np.append(x3,y3)

    polate1=np.append(x1,x2)
    polate1y=np.append(y1,y2)
    polate2=np.append(x2,x3)
    polate2y=np.append(y2,y3)
    polate3=np.append(x1,x3)
    polate3y=np.append(y1,y3)
    #print(poin1,poin2,poin3)
    for i in range(256):
        for j in range(256):
            
            if i<poin1[0] or i>poin3[0]: continue
            #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
            #print(poin1,poin3,np.array([i]),np.interp(np.array([i]),poin1,poin3))
            #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))            
            
            #if i<=poin2[0] and j<=np.interp(np.array([i]),poin1,poin2) and j>=np.interp(np.array([i]),poin1,poin3):
            if i<=poin2[0] and j<=np.interp(i,polate1,polate1y) and j>=np.interp(i,polate3,polate3y):
                #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
                #print(poin1,poin2,i,np.interp(i,poin1,poin2))
                #print(poin1,poin3,i,np.interp(i,poin1,poin3))
                #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(poin3[0]-i)
                b[i][j]=64+(z-64)*(poin3[0]-i)
            
            #elif i<=poin3[0] and i>poin2[0] and j<=np.interp(np.array([i]),poin2,poin3) and j>=np.interp(np.array([i]),poin1,poin3):
            elif i<=poin3[0] and i>poin2[0] and j<=np.interp(i,polate2,polate2y) and j>=np.interp(i,polate3,polate3y):
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(poin3[0]-i)
                b[i][j]=64+(z-64)*(poin3[0]-i)
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
           
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

#tipped triangle
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    p1=np.random.randint(0,20)
    p2=np.random.randint(0,20)
    q1=np.random.randint(20,256*(p1+p2)-21)
    q2=np.random.randint(q1+20,256*(p1+p2))
    
    
    x1=np.random.randint(10,229)
    x2=np.random.randint(x1+10,243)
    x3=np.random.randint(x2+10,255)

    y1=np.random.randint(20,243)
    y2=np.random.randint(y1+10,255)
    y3=np.random.randint(10,y1-9)
    
    poin1=np.append(x1,y1)
    poin2=np.append(x2,y2)
    poin3=np.append(x3,y3)

    polate1=np.append(x1,x2)
    polate1y=np.append(y1,y2)
    polate2=np.append(x2,x3)
    polate2y=np.append(y2,y3)
    polate3=np.append(x1,x3)
    polate3y=np.append(y1,y3)
    #print(poin1,poin2,poin3)
    for i in range(256):
        for j in range(256):
            
            if i<poin1[0] or i>poin3[0]: continue
            #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
            #print(poin1,poin3,np.array([i]),np.interp(np.array([i]),poin1,poin3))
            #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))            
            
            #if i<=poin2[0] and j<=np.interp(np.array([i]),poin1,poin2) and j>=np.interp(np.array([i]),poin1,poin3):
            if i<=poin2[0] and j<=np.interp(i,polate1,polate1y) and j>=np.interp(i,polate3,polate3y):
                #print(poin1,poin2,np.array([i]),np.interp(np.array([i]),poin1,poin2))
                #print(poin1,poin2,i,np.interp(i,poin1,poin2))
                #print(poin1,poin3,i,np.interp(i,poin1,poin3))
                #print(poin2,poin3,np.array([i]),np.interp(np.array([i]),poin2,poin3))
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(poin2[0]-i)
                b[i][j]=64+(z-64)*(poin2[0]-i)
            
            #elif i<=poin3[0] and i>poin2[0] and j<=np.interp(np.array([i]),poin2,poin3) and j>=np.interp(np.array([i]),poin1,poin3):
            elif i<=poin3[0] and i>poin2[0] and j<=np.interp(i,polate2,polate2y) and j>=np.interp(i,polate3,polate3y):
                if random.random()>0.6 :
                    a[i][j]=64+(z-64)*(i-poin2[0])
                b[i][j]=64+(z-64)*(i-poin2[0])
                
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

#rectangular roof
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    x1=np.random.randint(6,253)
    x2=np.random.randint(x1+2,255)

    y1=np.random.randint(6,253)
    y2=np.random.randint(y1+2,255)
    
    xav=(x1+x2)/2
    yav=(y1+y2)/2
    
    for i in range(256):
        for j in range(256):
            if i>=x1 and i<=x2 and j>=y1 and j<=y2:
                if i<xav:
                    if random.random()>0.6:
                        a[i][j]=64+(z-64)*(i-x1)/xav
                    b[i][j]=64+(z-64)*(i-x1)/xav
                else:
                    if random.random()>0.6:
                        a[i][j]=64+(z-64)*(x2-i)/xav
                    b[i][j]=64+(z-64)*(x2-i)/xav
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])


#random
for i in range(siz):
    r=np.random.randint(8,255)
    r2=np.random.randint(6,r-1)

    c=np.random.randint(256,size=(2))
    z=np.random.randint(64+6,255)
    #print(z)
    #print(r,c)
    a=np.zeros((256,256))
    b=np.zeros((256,256))
    #print(a)
    x=np.arange(256)
    y=np.arange(256)
    
    x1=np.random.randint(6,253)
    x2=np.random.randint(x1+2,255)

    y1=np.random.randint(6,253)
    y2=np.random.randint(y1+2,255)
    
    xav=(x1+x2)/2
    yav=(y1+y2)/2
    
    for i in range(256):
        for j in range(256):
            if random.random()<0.01:
                v=np.random.randint(64,255)
                a[i][j]=v
                b[i][j]=v
                
    
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))

    a=a.transpose(1,0)
    b=b.transpose(1,0)
    #print(a.shape)
    samp=np.append(samp,a.astype('int'))
    result=np.append(result,b.astype('int'))
        
with open("input.csv", "a+") as f:
    np.savetxt(f, samp)
with open("output.csv", "a+") as f:
    np.savetxt(f, result)
samp=np.array([])
result=np.array([])

samp=samp.reshape(int(samp.shape[0]/256/256),256,256)
result=result.reshape(int(result.shape[0]/256/256),256,256)

print(samp.shape,result.shape)

ax1 = plt.subplot(211)
ax1.set_aspect('equal')

# equivalent but more general
ax1.pcolormesh(x,y,a)

ax2 = plt.subplot(212)
ax2.set_aspect('equal')
# add a subplot with no frame
ax2.pcolormesh(x,y,b)

plt.show()


with open("input.csv", "r") as f:
    samp=np.loadtxt(f)
with open("output.csv", "r") as f:
    result=np.loadtxt(f)
samp=samp.reshape(int(samp.shape[0]/256/256),256,256)
result=result.reshape(int(result.shape[0]/256/256),256,256)

print(samp.shape,result.shape)
samp=np.array([])
result=np.array([])
