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
interpo=load_model('interpNetwork.hdf5')
interpo.load_weights('interpNetwork_weight.hdf5')
#interpo.summary()
#raise

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
scan= scan[ (scan[:,0]/reso).argsort() ]
scan=scan.astype('int')

scan_gr=scan[scan[:,1]<=6]
scan=scan[scan[:,1]>6]
#print(scan)
cluster = DBSCAN(eps=6, min_samples=5).fit(scan)
#print(cluster.labels_[0])
#print(scan[0])

clustered=np.dstack([scan[:,0],scan[:,1],scan[:,2],cluster.labels_])
clustered=clustered.reshape(clustered.shape[1],4)
#print(clustered)
#raise

#select=np.array([])
for i in range(-1,np.max(cluster.labels_)):
    #print(i, clustered[clustered[:,3]==i].shape[0])
    select=np.array([])
    triang=np.array([])
    #print( np.max(clustered[clustered[:,3]==i],axis=0), np.min(clustered[clustered[:,3]==i],axis=0) )
    if clustered[clustered[:,3]==i].shape[0]<100 or i==-1:
        if clustered[clustered[:,3]==i].shape[0]<=1: continue
        if select.shape[0]==0:
            select=clustered[clustered[:,3]==i][:,:-1]
            print(select.shape)
            #raise
        else:
            select=np.append(select, clustered[clustered[:,3]==i][:,:-1] )        
        continue
    
    samp=clustered[clustered[:,3]==i][:,:-1]
    #print("samp ", i)
    #print(samp)
    ind = np.lexsort((samp[:,0],samp[:,1],samp[:,2]))
    print("ind",ind)
    sorted_samp=samp[ind[::-1]]
    #print(sorted_samp)
    
    #z-sort
    scount=0
    while 1:
        if sorted_samp.shape[0]==0: break
        try:
            ztop0=sorted_samp[  (sorted_samp[:,1]>=np.max(sorted_samp[:,1])-3  )]
        except:
            print("183",sorted_samp)
            raise
        sorted_samp=sorted_samp[  (sorted_samp[:,1]<np.max(sorted_samp[:,1])-3  )]
        selected=np.array([])
        if ztop0.shape[0]<5:
            if select.shape[0]==0:
                select=ztop0
            else:
                select=np.append(select, ztop0)
            continue
        #=====================================
        while 1:
            if ztop0.shape[0]==0: break
            '''
            try:
                ztop=ztop0[  (ztop0[:,1]>=np.max(ztop0[:,1])-30  )]
            except:
                print("182",ztop0)
                raise
            ztop0=ztop0[  (ztop0[:,1]<np.max(ztop0[:,1])-30  )]
            selected=np.array([])
            '''
            cluster2 = DBSCAN(eps=6, min_samples=5).fit(ztop0)
            referp=np.argmax(ztop0, axis=0)[1]
            #print(np.argmax(ztop0, axis=0))
            #raise
            ztopl=cluster2.labels_[referp]
            #print(ztopl)
            #raise
            if ztopl==-1:
                
                if select.shape[0]==0:
                    select=np.expand_dims(ztop0[referp],axis=0)
                else:
                    select=np.append(select, np.expand_dims(ztop0[referp],axis=0) )
                #print(ztop0)
                ztop0=np.delete(ztop0,referp,axis=0)
                #print(ztop0)
                #raise
                continue
            #print(ztop0[ np.where(cluster2.labels_ == ztopl) ])
            #raise
            ztop=ztop0[ np.where(cluster2.labels_ == ztopl) ]
            ztop0=ztop0[ np.where(cluster2.labels_ != ztopl)]
            if ztop.shape[0]<5:
                if select.shape[0]==0:
                    select=ztop
                else:
                    select=np.append(select, ztop)
                continue

            #=====================================
            
            breakup=0
            try:
                Hull=ConvexHull(ztop) #, qhull_options="Qc")
            except:
                if select.shape[0]==0:
                    select=ztop
                else:
                    select=np.append(select, ztop)
                breakup=1
                pass
            if breakup==1:
                continue
                
            #print("912",Hull.simplices)
            #print(Hull.points)
            #raise
            #print("9233",Hull.coplanar)
            #if Hull.coplanar.shape[0]!=0: raise
            for simplex in Hull.simplices:
                #print(Hull.points[simplex][ 0], Hull.points[simplex][ 1], Hull.points[simplex][ 2])
                if selected.shape[0]==0:
                    selected=np.expand_dims(Hull.points[simplex][ 0],axis=0)
                    selected=np.append(selected,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    selected=np.append(selected,np.expand_dims(Hull.points[simplex][ 2],axis=0))              
                else:
                    selected=np.append(selected,np.expand_dims(Hull.points[simplex][ 0],axis=0))
                    selected=np.append(selected,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    selected=np.append(selected,np.expand_dims(Hull.points[simplex][ 2],axis=0))
            selected=selected.reshape(int(selected.shape[0]/3),3)
            #print("923",selected)
            selected=np.unique(selected, axis=0)
            #print(selected)
            if selected.shape[0]<5:
                for kk in range(ztop.shape[0]):
                    #print(np.expand_dims(ztop[kk],axis=0))
                    selected=np.append(selected,np.expand_dims(ztop[kk],axis=0),axis=0)
                    selected=np.unique(selected, axis=0)
                    #selected=selected.reshape(int(selected.shape[0]/3),3)
                    #print(selected)
                    
                    if selected.shape[0]>=5: break
            #print(selected)
            if select.shape[0]==0:
                select=selected
            else:
                select=np.append(select, selected)

            #raise
            pointer=5
            breakup=0
            while 1:
                if pointer>selected.shape[0]:
                    breakup=1
                    pointer=selected.shape[0]
                try:
                    mina=np.min(selected[pointer-5:pointer],axis=0)
                except:
                    #print("9233",pointer,selected)
                    raise
                maxa=np.max(selected[pointer-5:pointer],axis=0)
                norma=selected[0:5]-np.min(selected[pointer-5:pointer],axis=0)
                reso=maxa-mina
                reso[reso==0]=1
                norma/=reso
                #print("987",norma)
                
                #raise
                diff=np.diff(np.expand_dims(norma,axis=0),axis=1) #.astype('float')
                #print(diff)
                diff[diff==0]=0.00001
                #print(selected.shape,diff)

                grad_xy=diff[:,:,0]/diff[:,:,1]
                grad_xz=diff[:,:,0]/diff[:,:,2]
                grad_yz=diff[:,:,1]/diff[:,:,2]

                grad_xy[grad_xy<-1]=-2
                grad_xz[grad_xz<-1]=-2
                grad_yz[grad_yz<-1]=-2
                grad_xy[grad_xy>1]=2
                grad_xz[grad_xz>1]=2
                grad_yz[grad_yz>1]=2

                grad_xy/=2
                grad_xz/=2
                grad_yz/=2

                inpu=np.expand_dims(norma,axis=0).reshape(1,15)
                inpu2=inpu[:,0:12]
                diff=diff.reshape(diff.shape[0],12)

                outp=interpo.predict([inpu2,diff,grad_xy,grad_xz,grad_yz])
                outp=outp.reshape(int(outp.shape[1]/3),3)
                #print("999",outp)
                outp=outp*reso+mina
                outp=outp.astype('int')
                #print("999",outp)
                select=np.append(select, outp)

                if breakup==1: break
                pointer+=5
                    
            #print("972",select)
            #scount+=1
            #if scount==2: raise
            #raise
    select=select.reshape(int(select.shape[0]/3),3)
    select=np.unique(select, axis=0)
    selec=select
    #print(selec)
    cluste = DBSCAN(eps=6, min_samples=5, metric='manhattan').fit(selec)
    print(np.min(cluste.labels_))
    print(np.max(cluste.labels_))
    for jj in range( np.max(cluste.labels_)):
        if selec[np.where(cluste.labels_==jj)].shape[0]<5: continue
        coplanar=selec[np.where(cluste.labels_==jj)]

        cop=coplanar
        for pt in range(coplanar.shape[0]):
            
            copref=coplanar[pt]
            cop=np.roll(coplanar,pt)
            #print(copref)
            cop-=copref
            cop1=np.roll(cop,-1,axis=0)
            for pt1 in range(coplanar.shape[0]-1):
                cop2=np.roll(cop,-2,axis=0)
                for pt2 in range(coplanar.shape[0]-2):
                    cop3=np.roll(cop,-3,axis=0)
                    for pt3 in range(coplanar.shape[0]-3):
                        v1=cop1[0]
                        v2=cop2[0]
                        v3=cop3[0]
                        v=np.dot(v1,np.cross(v2,v3))
                        if v<0:
                            v=-v
                        if v<1:
                            #print(v,cop[0],cop1[0],cop2[0],cop3[0])
                            triang=np.append(triang,cop[0]+copref)
                            triang=np.append(triang,cop1[0]+copref)
                            triang=np.append(triang,cop2[0]+copref)
                            triang=np.append(triang,cop3[0]+copref)
                        cop3=np.roll(cop3,-1,axis=0)
                    triang=triang.reshape(int(triang.shape[0]/12),4,3)
                    triang=np.unique(triang,axis=1)
                if 1:
                    plt.figure()
                    custom=plt.subplot(121,projection='3d')

                    for tt in range(triang.shape[0]):
                        print(triang[tt],np.split(triang[tt],3,axis=1))
                        x,y,z=np.split(triang[tt],3,axis=1)
                        custom.scatter(x,z,y)
                        verts = [list(zip(x,z,y))]
                        srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
                        plt.gca().add_collection3d(srf)
                        #break
                        #raise
                    #x,y,z=np.split(select,3,axis=1)
                    #custom.scatter(x,z,y)
                        
                    plt.show()
                    cop2=np.roll(cop2,-1,axis=0)
                    raise
                    
                cop1=np.roll(cop1,-1,axis=0)
                
            cop=np.roll(cop,-1,axis=0)
            #raise
        '''
        #print(jj)
        Hull=ConvexHull(selec[np.where(cluste.labels_==jj)], qhull_options="QJ")
        #print(jj,Hull.points)
        if Hull.simplices.shape[0]>0:
            for simplex in Hull.simplices: #[np.unique(Hull.coplanar[:,1],axis=0)]:
                #print(Hull.points[simplex][ 0], Hull.points[simplex][ 1], Hull.points[simplex][ 2])
                if np.max( np.sum(np.abs(np.diff(Hull.points[simplex],axis=0)),axis=1) )>50: continue
                #print("933",Hull.points[simplex])
                if triang.shape[0]==0:
                    triang=np.expand_dims(Hull.points[simplex][ 0],axis=0)
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))              
                    #triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
                else:
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 0],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))
                    #triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))
        '''
    if 1:
        if 1:
            triang=triang.reshape(int(triang.shape[0]/12),4,3)
            plt.figure()
            custom=plt.subplot(121,projection='3d')

            for tt in range(triang.shape[0]):
                print(triang[tt],np.split(triang[tt],3,axis=1))
                x,y,z=np.split(triang[tt],3,axis=1)
                custom.scatter(x,z,y)
                verts = [list(zip(x,z,y))]
                srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
                plt.gca().add_collection3d(srf)
                #raise
            x,y,z=np.split(select,3,axis=1)
            custom.scatter(x,z,y)
                
            plt.show()
            raise
    '''
    while 1:
        refer=np.max(selec[:,1])
        selecte=selec[selec[:,1]>=refer-10]
        selec=selec[selec[:,1]<refer-10]

        cluste = DBSCAN(eps=15, min_samples=5, metric='manhattan').fit(selecte)
        
        for jj in range( np.max(cluste.labels_)):
            if cluste.labels_[cluste.labels_==jj].shape[0]<5: continue
            print(jj, cluste.labels_[cluste.labels_==jj].shape[0])
            #print(np.where(cluste.labels_==jj))
            #Hull=ConvexHull(select, qhull_options="QcQVn")
            continuee=0
            try:
                #Hull=Delaunay(selecte[np.where(cluste.labels_==jj)], qhull_options="Qx")
                Hull=ConvexHull(selecte[np.where(cluste.labels_==jj)], qhull_options="Q12")
            except:
                continuee=1
                pass
            if continuee==1: continue
            #print("972",select.shape,np.unique(Hull.simplices,axis=0).shape,Hull.coplanar)
            #print("972",np.unique(Hull.coplanar[:,1],axis=0), Hull.simplices[np.unique(Hull.coplanar[:,1],axis=0)])
            #print("972",Hull)

            
            #print("972",select)
            #raise
            #triang=np.array([])
            for simplex in Hull.simplices: #[np.unique(Hull.coplanar[:,1],axis=0)]:
                #print(Hull.points[simplex][ 0], Hull.points[simplex][ 1], Hull.points[simplex][ 2])
                #if np.max( np.sum(np.abs(np.diff(Hull.points[simplex],axis=0)),axis=1) )>15: continue
                print("933",Hull.points[simplex])
                if triang.shape[0]==0:
                    triang=np.expand_dims(Hull.points[simplex][ 0],axis=0)
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))              
                    #triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
                else:
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 0],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))
                    #triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
            triang=triang.reshape(int(triang.shape[0]/9),3,3)
            #triang=triang.reshape(int(triang.shape[0]/12),4,3)
            #print(triang)
            #raise
        if selec.shape[0]<5: break
    
    selec=select
    while 1:
        refer=np.max(selec[:,2])
        selecte=selec[selec[:,2]>=refer-10]
        selec=selec[selec[:,2]<refer-10]

        cluste = DBSCAN(eps=10, min_samples=5, metric='manhattan').fit(selecte)
        
        for jj in range( np.max(cluste.labels_)):
            if cluste.labels_[cluste.labels_==jj].shape[0]<5: continue
            #print(np.where(cluste.labels_==jj))
            #Hull=ConvexHull(select, qhull_options="QcQVn")
            continuee=0
            try:
                Hull=Delaunay(selecte[np.where(cluste.labels_==jj)]) #, qhull_options="QcQVn")
            except:
                continuee=1
                pass
            if continuee==1: continue
            #print("972",select.shape,np.unique(Hull.simplices,axis=0).shape,Hull.coplanar)
            #print("972",np.unique(Hull.coplanar[:,1],axis=0), Hull.simplices[np.unique(Hull.coplanar[:,1],axis=0)])
            #print("972",Hull)

            
            #print("972",select)
            #raise
            #triang=np.array([])
            for simplex in Hull.simplices: #[np.unique(Hull.coplanar[:,1],axis=0)]:
                #print(Hull.points[simplex][ 0], Hull.points[simplex][ 1], Hull.points[simplex][ 2])
                if np.max( np.sum(np.abs(np.diff(Hull.points[simplex],axis=0)),axis=1) )>15: continue
                #print("933",simplex,Hull.points.shape)
                if triang.shape[0]==0:
                    triang=np.expand_dims(Hull.points[simplex][ 0],axis=0)
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))              
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
                else:
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 0],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
            #triang=triang.reshape(int(triang.shape[0]/9),3,3)
            triang=triang.reshape(int(triang.shape[0]/12),4,3)
            print(triang)
            #raise
        if selec.shape[0]<5: break

    selec=select
    while 1:
        refer=np.max(selec[:,0])
        selecte=selec[selec[:,0]>=refer-10]
        selec=selec[selec[:,0]<refer-10]

        cluste = DBSCAN(eps=10, min_samples=5, metric='manhattan').fit(selecte)
        
        for jj in range( np.max(cluste.labels_)):
            if cluste.labels_[cluste.labels_==jj].shape[0]<5: continue
            #print(np.where(cluste.labels_==jj))
            #Hull=ConvexHull(select, qhull_options="QcQVn")
            continuee=0
            try:
                Hull=Delaunay(selecte[np.where(cluste.labels_==jj)], qhull_options="Q2")
            except:
                #Hull=Delaunay(selecte[np.where(cluste.labels_==jj)], qhull_options="Q2Qbk:0Bk:0")
                #print(i m here)"
                continuee=1
                pass
            if continuee==1: continue
            #print("972",select.shape,np.unique(Hull.simplices,axis=0).shape,Hull.coplanar)
            #print("972",np.unique(Hull.coplanar[:,1],axis=0), Hull.simplices[np.unique(Hull.coplanar[:,1],axis=0)])
            #print("972",Hull)

            
            #print("972",select)
            #raise
            #triang=np.array([])
            for simplex in Hull.simplices: #[np.unique(Hull.coplanar[:,1],axis=0)]:
                #print(Hull.points[simplex][ 0], Hull.points[simplex][ 1], Hull.points[simplex][ 2])
                #print("933",simplex,Hull.points.shape)
                #print("933",Hull.points[simplex],np.diff(Hull.points[simplex],axis=0),np.sum(np.abs(np.diff(Hull.points[simplex],axis=0)),axis=1) )
                #if np.max( np.sum(np.abs(np.diff(Hull.points[simplex],axis=0)),axis=1) )>15: continue
                #print("998")
                if triang.shape[0]==0:
                    triang=np.expand_dims(Hull.points[simplex][ 0],axis=0)
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))              
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
                else:
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 0],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 1],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 2],axis=0))
                    triang=np.append(triang,np.expand_dims(Hull.points[simplex][ 3],axis=0))              
            triang=triang.reshape(int(triang.shape[0]/12),4,3)
            #triang=triang.reshape(int(triang.shape[0]/9),3,3)
            print(triang)
            #raise
        if selec.shape[0]<5: break
        '''

    X=triang.transpose(0,2,1)[:,0]  
    Y=triang.transpose(0,2,1)[:,1]  
    Z=triang.transpose(0,2,1)[:,2]
    for i in range(X.shape[0]):
        if i<=40 and i>29:
            print("200",i,X[i],Y[i],Z[i])
        if np.max(Y[i])-np.max(Y[i])>10:
            print("200",i,X[i],Y[i],Z[i])
        if np.max(X[i])-np.max(X[i])>10:
            print("200",i,X[i],Y[i],Z[i])
        if np.max(Z[i])-np.max(Z[i])>10:
            print("200",i,X[i],Y[i],Z[i])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title("buildings")

    #ax.plot_surface(X, Z, Y,cmap='Reds', edgecolor='none')

    '''
    for ii in range(X.shape[0]):
        kk=ii+1
        #print(X[ii:kk], Z[ii:kk], Y[ii:kk])
        ax.plot_surface(X[ii:kk], Z[ii:kk], Y[ii:kk], rstride=1, cstride=1,edgecolor='none')
    '''
    ax.plot_surface( np.array([[1,0,1,8],[5,6,5,10]]), np.array([[1,1,3,12],[5,5,8,15]]), np.array([[3,1,1,19],[8,5,5,20]]) )
    x,y,z=np.split(select,3,axis=1)
    #ax.scatter(x, z, y, zdir='z', c= 'red')
    
    fig.show()
    raise
