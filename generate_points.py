import os,re
import numpy as np
from scipy.interpolate import LinearNDInterpolator

reso=32
def find_xy(p1, p2, z):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if z1==z2:
        #print(x1,y1,y2)
        if x1==x2: return x2,y1+(y2-y1)/2
        return x1+(x2-x1)/2,y1+(y2-y1)/2

    if z2 < z1:
        return find_xy(p2, p1, z)

    x = np.interp(z, (z1, z2), (x1, x2))
    y = np.interp(z, (z1, z2), (y1, y2))

    return x, y

def find_yz(p1, p2, x):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if x1==x2:
        #print(x1,y1,y2)
        if y1==y2: return y2,z1+(z2-z1)/2
        return y1+(y2-y1)/2,z1+(z2-z1)/2
    if x2 < x1:
        return find_yz(p2, p1, x)

    z = np.interp(x, (x1, x2), (z1, z2))
    y = np.interp(x, (x1, x2), (y1, y2))

    return y, z

def find_xz(p1, p2, y):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if y1==y2:
        #print(x1,y1,y2)
        if z1==z2: return x1+(x2-x1)/2,z2
        return x1+(x2-x1)/2,z1+(z2-z1)/2
    if y2 < y1:
        return find_xz(p2, p1, y)

    z = np.interp(y, (y1, y2), (z1, z2))
    x = np.interp(y, (y1, y2), (x1, x2))

    return x, z


#5-points straight 
for s in range(2):
    
    for i in range(reso):
        while 1:
            #a=np.round(1+29*np.random.rand(7,3)).astype('int')
            temp=np.sort( np.random.rand(5) )
            temp-=temp[0]
            r=temp/np.max(temp)
            
            #print(r,np.sum(r))
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base[2]=0
            #print(base)
            theta=(s)*2.5/180*3.1416
            
            
            maxx=max(base[0],reso-base[0])*np.sin(theta)
            maxy=max(base[1],reso-base[1])*np.cos(theta)
            maxz=max(base[2],reso-base[2])
            #print(maxx,maxy,maxz)
            #print(( maxx**2+maxy**2 )**0.5)
            rr=(0.2+0.8*np.random.rand())*(( maxx**2+maxy**2 )**0.5)
            r*=rr
            #print(rr,r)
            if r[1]-r[0]>=2 and r[2]-r[1]>=2 and r[3]-r[2]>=2 and r[4]-r[3]>=2: break
        
        x= r*np.sin(theta)
        y= r*np.cos(theta)
        if base[0]>reso/2:
            if base[1]>reso/2:
                x=base[0]-x
                y=base[1]-y
                z=np.zeros(5)
            else:
                x=base[0]-x
                y=base[1]+y
                z=np.zeros(5)
        else:
            if base[1]>reso/2:
                x=base[0]+x
                y=base[1]-y
                z=np.zeros(5)
            else:
                x=base[0]+x
                y=base[1]+y
                z=np.zeros(5)
            
        points=np.dstack([x,y,z]).reshape(5,3)
        #print(points)
        points.astype('int')

        
        for j in range(4):
            diff=points[j+1]-points[j]
            median=np.round(diff[0]/2)
            ym,zm= find_yz(points[j],points[j+1],points[j][0]+median) 
            bb=np.array([[points[j][0]+median,ym,zm]]).astype('int')
            #print(bb)
            if j==0:
                b=bb
            else:
                b=np.append(b,bb,axis=0)
        #print(b)

        #raise
         
        
        #print( find_xy(p1, p2, z) )
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")
    
    for i in range(reso):
        while 1:
            #a=np.round(1+29*np.random.rand(7,3)).astype('int')
            temp=np.sort( np.random.rand(5) )
            temp-=temp[0]
            r=temp/np.max(temp)
            
            #print(r,np.sum(r))
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base[1]=0
            #print(base)
            theta=(s)*2.5/180*3.1416
            
            
            maxx=max(base[0],reso-base[0])*np.sin(theta)
            maxy=max(base[1],reso-base[1])
            maxz=max(base[2],reso-base[2])*np.cos(theta)
            #print(maxx,maxy,maxz)
            #print(( maxx**2+maxy**2 )**0.5)
            rr=(0.2+0.8*np.random.rand())*(( maxx**2+maxz**2 )**0.5)
            r*=rr
            #print(rr,r)
            if r[1]-r[0]>=3 and r[2]-r[1]>=3 and r[3]-r[2]>=3 and r[4]-r[3]>=3: break
        
        x= r*np.sin(theta)
        z= r*np.cos(theta)
        if base[0]>reso/2:
            if base[2]>reso/2:
                x=base[0]-x
                z=base[2]-z
                y=np.zeros(5)
            else:
                x=base[0]-x
                z=base[2]+z
                y=np.zeros(5)
        else:
            if base[2]>reso/2:
                x=base[0]+x
                z=base[2]-z
                y=np.zeros(5)
            else:
                x=base[0]+x
                z=base[2]+z
                y=np.zeros(5)
            
        points=np.dstack([x,y,z]).reshape(5,3)
        #print(points)
        points.astype('int')
        
        for j in range(4):
            diff=points[j+1]-points[j]
            median=np.round(diff[2]/2)
            xm, ym= find_xy(points[j],points[j+1],points[j][2]+median) 
            bb=np.array([[ xm,ym,points[j][2]+median]]).astype('int')
            #print(bb)
            if j==0:
                b=bb
            else:
                b=np.append(b,bb,axis=0)
        #print(b)

        #raise
         
        
        #print( find_xy(p1, p2, z) )
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")
        
    
    for i in range(reso):
        while 1:
            #a=np.round(1+29*np.random.rand(7,3)).astype('int')
            temp=np.sort( np.random.rand(5) )
            temp-=temp[0]
            r=temp/np.max(temp)
            
            #print(r,np.sum(r))
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base[0]=0
            #print(base)
            theta=(s)*2.5/180*3.1416
            
            
            maxx=max(base[0],reso-base[0])
            maxy=max(base[1],reso-base[1])*np.sin(theta)
            maxz=max(base[2],reso-base[2])*np.cos(theta)
            #print(maxx,maxy,maxz)
            #print(( maxx**2+maxy**2 )**0.5)
            rr=(0.2+0.8*np.random.rand())*(( maxy**2+maxz**2 )**0.5)
            r*=rr
            #print(rr,r)
            if r[1]-r[0]>=2 and r[2]-r[1]>=2 and r[3]-r[2]>=2 and r[4]-r[3]>=2: break
        
        y= r*np.sin(theta)
        z= r*np.cos(theta)
        if base[1]>reso/2:
            if base[2]>reso/2:
                y=base[1]-y
                z=base[2]-z
                x=np.zeros(5)
            else:
                y=base[1]-y
                z=base[2]+z
                x=np.zeros(5)
        else:
            if base[2]>reso/2:
                y=base[1]+y
                z=base[2]-z
                x=np.zeros(5)
            else:
                y=base[1]+y
                z=base[2]+z
                x=np.zeros(5)
            
        points=np.dstack([x,y,z]).reshape(5,3)
        #print(points)
        points.astype('int')
        
        for j in range(4):
            diff=points[j+1]-points[j]
            median=np.round(diff[1]/2)
            xm, zm= find_xz(points[j],points[j+1],points[j][1]+median) 
            bb=np.array([[ xm,points[j][1]+median,zm]]).astype('int')
            #print(bb)
            if j==0:
                b=bb
            else:
                b=np.append(b,bb,axis=0)
        #print(b)

        #raise
         
        
        #print( find_xy(p1, p2, z) )
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")
'''    

#circle
for s in range(36):
    
    for i in range(reso):
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise

        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            x=r*np.sin(theta)
            y=r*np.cos(theta)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            z=i*np.ones(5)
        
            if base[0]>reso/2:
                if base[1]>reso/2:
                    x=base[0]-x
                    y=base[1]-y
                else:
                    x=base[0]-x
                    y=base[1]+y
            else:
                if base[1]>reso/2:
                    x=base[0]+x
                    y=base[1]-y
                else:
                    x=base[0]+x
                    y=base[1]+y

            
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue

            #print(np.diff(theta))
            theta2=theta[:-1]+np.diff(theta)/2
            #print(theta,theta2)
            x=r*np.sin(theta2)
            y=r*np.cos(theta2)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            z=i*np.ones(4)
        
            if base[0]>reso/2:
                if base[1]>reso/2:
                    x=base[0]-x
                    y=base[1]-y
                else:
                    x=base[0]-x
                    y=base[1]+y
            else:
                if base[1]>reso/2:
                    x=base[0]+x
                    y=base[1]-y
                    z=np.zeros(4)
                else:
                    x=base[0]+x
                    y=base[1]+y
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise
        
    for i in range(reso):
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise

        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            x=r*np.sin(theta)
            z=r*np.cos(theta)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            y=i*np.ones(5)
        
            if base[0]>reso/2:
                if base[2]>reso/2:
                    x=base[0]-x
                    z=base[2]-z
                else:
                    x=base[0]-x
                    z=base[2]+z
            else:
                if base[2]>reso/2:
                    x=base[0]+x
                    z=base[2]-z
                else:
                    x=base[0]+x
                    z=base[2]+z
            
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue

            #print(np.diff(theta))
            theta2=theta[:-1]+np.diff(theta)/2
            #print(theta,theta2)
            x=r*np.sin(theta2)
            z=r*np.cos(theta2)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            y=i*np.ones(4)
        
            if base[0]>reso/2:
                if base[2]>reso/2:
                    x=base[0]-x
                    z=base[2]-z
                else:
                    x=base[0]-x
                    z=base[2]+z
            else:
                if base[2]>reso/2:
                    x=base[0]+x
                    z=base[2]-z
                else:
                    x=base[0]+x
                    z=base[2]+z
            
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise
        
    for i in range(reso):
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise

        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            y=r*np.sin(theta)
            z=r*np.cos(theta)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            x=i*np.ones(5)
        
            if base[1]>reso/2:
                if base[2]>reso/2:
                    y=base[1]-y
                    z=base[2]-z
                else:
                    y=base[1]-y
                    z=base[2]+z
            else:
                if base[2]>reso/2:
                    y=base[1]+y
                    z=base[2]-z
                else:
                    y=base[1]+y
                    z=base[2]+z
            
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue

            #print(np.diff(theta))
            theta2=theta[:-1]+np.diff(theta)/2
            #print(theta,theta2)
            y=r*np.sin(theta2)
            z=r*np.cos(theta2)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            x=i*np.ones(4)
        
            if base[1]>reso/2:
                if base[2]>reso/2:
                    y=base[1]-y
                    z=base[2]-z
                else:
                    y=base[1]-y
                    z=base[2]+z
            else:
                if base[2]>reso/2:
                    y=base[1]+y
                    z=base[2]-z
                else:
                    y=base[1]+y
                    z=base[2]+z
            
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise


#5-points straight 
for s in range(36):
    
    for i in range(36):
        while 1:
            #a=np.round(1+29*np.random.rand(7,3)).astype('int')
            temp=np.sort( np.random.rand(5) )
            temp-=temp[0]
            r=temp/np.max(temp)
            
            #print(r,np.sum(r))
            base=np.round(1+29*np.random.rand(3)).astype('int')
            
            #print(base)
            theta=(s)*2.5/180*3.1416
            thetab=(i)*2.5/180*3.1416
            
            
            maxx=max(base[0],reso-base[0])*np.sin(theta)*np.sin(thetab)
            maxy=max(base[1],reso-base[1])*np.cos(theta)*np.sin(thetab)
            maxz=max(base[2],reso-base[2])*np.cos(thetab)
            #print(maxx,maxy,maxz)
            #print(( maxx**2+maxy**2 )**0.5)
            rr=(0.2+0.8*np.random.rand())*(( maxx**2+maxy**2+maxz**2 )**0.5)
            r*=rr
            #print(rr,r)
            if r[1]-r[0]<=2 or r[2]-r[1]<=2 or r[3]-r[2]<=2 or r[4]-r[3]<=2: continue
        
            x= r*np.sin(theta)*np.sin(thetab)
            y= r*np.cos(theta)*np.sin(thetab)
            z= r*np.cos(thetab)
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z
                
                
            points=np.dstack([x,y,z]).reshape(5,3)
            if np.max(points)>=reso or np.min(points)<0: continue
            #print(points)
            #print(np.diff(points,axis=0))
            b=points[:-1]+np.diff(points,axis=0)/2
            #print(b)
            points.astype('int')
            
            break

                 
        
        #print( find_xy(p1, p2, z) )
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

#circle
for s in range(72):
    
    for i in range(36):
        print(s,i)
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise
        thetab=(i)*2.5/180*3.1416
        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
            if r<4: continue
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            x=r*np.sin(theta)
            y=r*np.cos(theta)*np.cos(thetab)
            #print(i,r,thetab,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            z=r*np.cos(theta)*np.sin(thetab)*np.ones(5)
            #print("r",r,z,np.sin(thetab).shape)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z

            #print(x,y,z)
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue
            #print(x,y,z)
            theta2=theta[:-1]+np.diff(theta)/2
            x=r*np.sin(theta2)
            y=r*np.cos(theta2)*np.cos(thetab)
            z=r*np.cos(theta2)*np.ones(4)*np.sin(thetab)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z
            #print(x,y,z)
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        if 0:

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, zdir='z', c= 'red')

            plt.savefig("demo.png")
            plt.show()
        
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise

          
#circle
for s in range(72):
    
    for i in range(36):
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise
        thetab=(i)*2.5/180*3.1416
        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
            if r<4: continue
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            x=r*np.sin(theta)
            z=r*np.cos(theta)*np.cos(thetab)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            y=r*np.cos(theta)*np.sin(thetab)*np.ones(5)
            #print("r",r,z,np.sin(thetab).shape)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z

            #print(x,y,z)
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue
            
            theta2=theta[:-1]+np.diff(theta)/2
            x=r*np.sin(theta2)
            z=r*np.cos(theta2)*np.cos(thetab)
            y=r*np.cos(theta2)*np.ones(4)*np.sin(thetab)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z
            #print(x,y,z)
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        if 0:

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, zdir='z', c= 'red')

            plt.savefig("demo.png")
            plt.show()
        
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise


for s in range(72):
    
    for i in range(36):
        base=np.round(1+29*np.random.rand(3)).astype('int')
        rmax=max(np.max(reso-base) , np.max(base))-1
        #raise
        thetab=(i)*2.5/180*3.1416
        
        while 1:
            r=4+(rmax-4)*np.random.rand(1)        
            if r>reso-3: r=reso-3
            if r<4: continue
                
            theta=np.sort( 0.5*3.1416*np.random.rand(5) )
            if theta[1]-theta[0]<0.2 or theta[2]-theta[1]<0.2 or theta[3]-theta[2]<0.2 or theta[4]-theta[3]<0.2: continue
            #print(theta)
            z=r*np.sin(theta)
            y=r*np.cos(theta)*np.cos(thetab)
            #print(r,x,y)
            #if np.min(x)<0 or np.max(x)>=reso or np.min(y)<0 or np.max(y)>=reso:continue
            x=r*np.cos(theta)*np.sin(thetab)*np.ones(5)
            #print("r",r,z,np.sin(thetab).shape)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z

            #print(x,y,z)
            points=np.dstack([x,y,z]).reshape(5,3).astype('int')
            if np.min(points)<0 or np.max(points)>=reso: continue
            
            theta2=theta[:-1]+np.diff(theta)/2
            z=r*np.sin(theta2)
            y=r*np.cos(theta2)*np.cos(thetab)
            x=r*np.cos(theta2)*np.ones(4)*np.sin(thetab)
        
            if base[2]>reso/2:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]-z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]-z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]-z
            else:
                if base[0]>reso/2:
                    if base[1]>reso/2:
                        x=base[0]-x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]-x
                        y=base[1]+y
                        z=base[2]+z
                else:
                    if base[1]>reso/2:
                        x=base[0]+x
                        y=base[1]-y
                        z=base[2]+z
                    else:
                        x=base[0]+x
                        y=base[1]+y
                        z=base[2]+z
            #print(x,y,z)
            b=np.dstack([x,y,z]).reshape(4,3).astype('int')
            if np.min(b)<0 or np.max(b)>=reso: continue
                
            break
        #print(points)
        #print(b)
        if 0:

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, zdir='z', c= 'red')

            plt.savefig("demo.png")
            plt.show()
        
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

        
        #raise

for s in range(72):
    
    for i in range(36):
        while 1:
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base2=np.round(1+29*np.random.rand(3)).astype('int')
            temp= base2-base
            if np.min(temp)<2:continue
            prob=np.random.rand(1)
            if prob<0.33:
                temp[0]=-temp[0]
            elif prob<0.33:
                temp[1]=-temp[1]
            else:
                temp[2]=-temp[2]
            base3=base2+temp
            if np.min(base3)<0 or np.max(base3)>=reso: continue
            #print(base,temp)
            #print(base2)
            #print(base3)
            p1=base+((base2-base)/2)
            p1=p1.astype('int')
            p2=base2+((base3-base2)/2)
            p2=p2.astype('int')
            #print(p1,p2)
            points=np.stack([base,p1,base2,p2,base3])
            #print(points)
            b=points[:-1]+np.diff(points,axis=0)/2
            b=b.astype('int')
            #print(b)
            break

        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")
        #raise


for s in range(72):    
    for i in range(36):
        while 1:
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base2=np.round(1+29*np.random.rand(3)).astype('int')
            temp=base2-base
            #print(temp)
            prob=np.random.rand(1)
            if prob<0.33:
                temp[0]=-temp[0]
            elif prob<0.33:
                temp[1]=-temp[1]
            else:
                temp[2]=-temp[2]
            base3=base2+((0.5*np.random.rand()+0.5)*temp)
            if np.min(base3)<0 or np.max(base3)>=reso: continue
            #print(base,base2,base3)
            #raise

            temp=base3-base2
            prob=np.random.rand(1)
            if prob<0.33:
                temp[0]=-temp[0]
            elif prob<0.33:
                temp[1]=-temp[1]
            else:
                temp[2]=-temp[2]
            base4=base3+((0.5*np.random.rand()+0.5)*temp)
            if np.min(base4)<0 or np.max(base4)>=reso: continue
            #print(base,base2,base3,base4)
            #raise

            temp=base4-base3
            prob=np.random.rand(1)
            if prob<0.33:
                temp[0]=-temp[0]
            elif prob<0.33:
                temp[1]=-temp[1]
            else:
                temp[2]=-temp[2]
            base5=base4+((0.5*np.random.rand()+0.5)*temp)
            if np.min(base5)<0 or np.max(base5)>=reso: continue
            #print(base,base2,base3,base4,base5)

            points=np.stack([base,base2,base3.astype('int'),base4.astype('int'),base5.astype('int')])
            #print(points)
            b=points[:-1]+np.diff(points,axis=0)/2
            b=b.astype('int')
            #print(b)
            
            break
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")
        
        #raise


for s in range(72):    
    for i in range(36):
        print(s,i)
        while 1:
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base2=np.round(1+29*np.random.rand(3)).astype('int')
            temp=base2-base
            base3=base2+(0.5+np.random.rand()*0.5)*temp
            if np.min(base3)<0 or np.max(base3)>=reso: continue
            diff=np.abs(temp)
            if np.min(temp)<=2: continue
            #print(base,base2,base3)
            prob=np.random.rand()
            #print(prob)
            #print(temp)
            prev=base2-base
            if prob<0.33:
                #print(2+3*np.random.rand())
                temp[0]=temp[0]*(1+np.random.rand())
                temp[1]=temp[1]*(1+np.random.rand())
            elif prob<0.66:
                temp[1]=temp[1]*(1+np.random.rand())
                temp[2]=temp[2]*(1+np.random.rand())
            else:
                temp[0]=temp[0]*(1+np.random.rand())
                temp[2]=temp[2]*(1+np.random.rand())
            #print(temp,prev)
            #raise
            if np.max(np.abs(prev-temp))<=2: continue
            
            base4=base3+(0.5+np.random.rand()*0.5)*temp
            if np.min(base4)<0 or np.max(base4)>=reso: continue
            if prob<0.33:
                #print(2+3*np.random.rand())
                temp[0]=temp[0]*(np.random.rand()+np.random.rand())
                temp[1]=temp[1]*(np.random.rand()+np.random.rand())
            elif prob<0.66:
                temp[1]=temp[1]*(np.random.rand()+np.random.rand())
                temp[2]=temp[2]*(np.random.rand()+np.random.rand())
            else:
                temp[0]=temp[0]*(np.random.rand()+np.random.rand())
                temp[2]=temp[2]*(np.random.rand()+np.random.rand())
            
            base5=base4+(0.5+np.random.rand()*0.5)*temp
            if np.min(base5)<0 or np.max(base5)>=reso: continue
            #print(base,base2,base3,base4,base5)
            #print(base3-base2,base4-base3,(base3-base2)/(base4-base3))
            
            points=np.stack([base,base2,base3,base4,base5]).reshape(5,3)
            b=points[:-1]+np.diff(points,axis=0)
            #print(points,b)
            #raise
            
            break
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")


for s in range(72):    
    for i in range(36):
        while 1:
            base=np.round(1+29*np.random.rand(3)).astype('int')
            base2=np.round(1+29*np.random.rand(3)).astype('int')
            temp=base2-base
            base3=base2+(0.5+np.random.rand()*0.5)*temp
            if np.min(base3)<0 or np.max(base3)>=reso: continue
            diff=np.abs(temp)
            if np.min(temp)<=2: continue
            #print(base,base2,base3)
            prob=np.random.rand()
            #print(prob)
            #print(temp)
            prev=base2-base
            if prob<0.33:
                #print(2+3*np.random.rand())
                temp[0]=temp[0]*(1+np.random.rand())
                temp[1]=temp[1]*(1+np.random.rand())
            elif prob<0.66:
                temp[1]=temp[1]*(1+np.random.rand())
                temp[2]=temp[2]*(1+np.random.rand())
            else:
                temp[0]=temp[0]*(1+np.random.rand())
                temp[2]=temp[2]*(1+np.random.rand())
            #print(temp,prev)
            #raise
            if np.max(np.abs(prev-temp))<=2: continue
            
            base4=base3+(0.5+np.random.rand()*0.5)*temp
            if np.min(base4)<0 or np.max(base4)>=reso: continue
            if prob<0.33:
                #print(2+3*np.random.rand())
                temp[0]=temp[0]*(np.random.rand()+np.random.rand())
                temp[1]=temp[1]*(np.random.rand()+np.random.rand())
            elif prob<0.66:
                temp[1]=temp[1]*(np.random.rand()+np.random.rand())
                temp[2]=temp[2]*(np.random.rand()+np.random.rand())
            else:
                temp[0]=temp[0]*(np.random.rand()+np.random.rand())
                temp[2]=temp[2]*(np.random.rand()+np.random.rand())
            
            base5=base4+(0.5+np.random.rand()*0.5)*temp
            if np.min(base5)<0 or np.max(base5)>=reso: continue
            #print(base,base2,base3,base4,base5)
            #print(base3-base2,base4-base3,(base3-base2)/(base4-base3))
            
            points=np.stack([base,base2,base3,base4,base5]).reshape(5,3)
            points=points[::-1]
            b=points[:-1]+np.diff(points,axis=0)
            #print(points,b)
            #print(points[::-1])
            #raise
            
            break
        with open('input.csv','a+') as opener:
            aaa=points.reshape(points.shape[0]*3)
            np.savetxt(opener, aaa, newline=" ",fmt="%d", delimiter=",")
            opener.write("\n")

        with open('output.csv','a+') as opener2:
            bbb=b.reshape(b.shape[0]*3)
            np.savetxt(opener2, bbb, newline=" ",fmt="%d", delimiter=",")
            opener2.write("\n")

'''        

