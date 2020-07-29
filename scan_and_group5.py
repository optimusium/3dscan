import os,re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull,Delaunay

from tensorflow.keras.models import Model,load_model

def find_xy(p1, p2, z):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if z2 < z1:
        return find_xy(p2, p1, z)

    x = np.interp(z, (z1, z2), (x1, x2))
    y = np.interp(z, (z1, z2), (y1, y2))

    return x, y
model = load_model('map_network_classifier.hdf5')
model.load_weights("map_network_weight.hdf5")

inf0=open("scan166.pose","r")
pos0=inf0.readline().replace("\n","").replace(" ",",")
pos0=np.array(eval("["+pos0+"]"))
eu=inf0.readline().replace("\n","").replace(" ",",")
eu=np.array( eval("["+eu+"]")  )*math.pi/180
inf0.close()

print(eu)
print(math.cos(eu[1]))

matz=[[ math.cos(eu[2]) , -1*math.sin(eu[2]) , 0 ],[ math.sin(eu[2]) ,  1*math.cos(eu[2]) , 0 ],[ 0 , 0, 1] ]
maty=[[ math.cos(eu[1]) , 0, 1*math.sin(eu[1]) ],[ 0 , 1, 0],[ -1*math.sin(eu[1]) , 0, 1*math.cos(eu[1])] ]
matx=[ [1,0,0],[ 0, math.cos(eu[0]) , -1*math.sin(eu[0]) ],[ 0, math.sin(eu[0]) ,  1*math.cos(eu[0]) ] ]
matz=np.array(matz)
maty=np.array(maty)
matx=np.array(matx)

mat=matz*maty*matx
print(mat)
inf=open("scan166.3d","r")
buf=[]
buf2=[]
pos=inf.readline().replace("\n","").replace(" ",",")
pos=np.array(eval("["+pos+"]"))

for line in inf.readlines():
    line=line.replace("\n","").replace(" ",",")
    #print(line)
    p=eval("["+line+"]")
    buf.append(eval("["+line+"]"))
    
    posi=np.transpose( np.array(eval("["+line+"]"))-pos0 )
    res=np.round( np.matmul(mat,posi)+pos0 )
    #res=posi
    #print(p,pos,posi,res)
    buf2.append( res.tolist() )

    #print(buf,buf2)
    #break


inf.close()

pos0=np.round(pos0)

reso=32
scan=np.array(buf2)
scan= scan[ (scan[:,0]).argsort() ]
scan=scan.astype('int')

xmin=np.min(scan[:,0])
xmax=np.max(scan[:,0])
ymin=np.min(scan[:,2])
ymax=np.max(scan[:,2])
print(xmax,xmin,ymax,ymin)
nx=np.ceil( (xmax)/256 )
ny=np.ceil( (ymax)/256 )
print(nx,ny)
startx=math.floor(xmin/256)*256
starty=math.floor(ymin/256)*256
endx=math.floor(xmax/256)*256
endy=math.floor(ymax/256)*256

x=np.arange(256)
y=np.arange(256)
for ii in range(startx,endx+256,256):
    print("919",ii)
    print(scan[ (scan[:,0]>=ii) & (scan[:,0]<ii+256)].shape[0])
    for jj in range(starty,endy+256,256):
        print("909",ii,jj)
        #continue
        if scan[ (scan[:,0]>=ii) & (scan[:,0]<ii+256) & (scan[:,2]>=jj) & (scan[:,2]<jj+256)].shape[0]<=10:
            continue 
        print( scan[ (scan[:,0]>=ii) & (scan[:,0]<ii+256) & (scan[:,2]>=jj) & (scan[:,2]<jj+256)].shape[0] )
        
        map1=np.zeros((256,256))
        for i in range(256):
            for j in range(256):
                try:
                    map1[i][j]=32+np.max( scan[ (scan[:,0]==ii+i) & (scan[:,2]==jj+j) ][:,1] )
                except:
                    pass
                
        map2=map1
        map2-=32
        map2/=224        
        map3=map2.reshape(256,256,1)
        
        res_model=model.predict(np.expand_dims(map3,axis=0))
        ax1 = plt.subplot(211)
        ax1.set_aspect('equal')
        
        # equivalent but more general
        ax1.pcolormesh(x,y,map2)
        
        ax3 = plt.subplot(212)
        ax3.set_aspect('equal')
        # add a subplot with no frame
        ax3.pcolormesh(x,y,res_model[0][:,:,0].reshape(256,256))
        
        plt.show()
        #raise

                
