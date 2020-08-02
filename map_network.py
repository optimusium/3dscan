# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:43:52 2020

@author: boonping
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D,Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda,Concatenate,Multiply,Dot,Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
#from tensorflow.keras import backend



inp=Input(shape=(256,256,1,))
#negInp=Input(shape=(256,256,1,))
x1a=MaxPooling2D(pool_size=(16, 16), strides=(16,16))(inp)
x1a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x1a)
x1b=AveragePooling2D(pool_size=(16, 16), strides=(16,16))(inp)
x1b=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x1b)
x1a=Concatenate()([x1b,x1a])
x1c=Conv2D(16,kernel_size=(16,16), strides=(16,16),padding="same",activation='relu')(inp)
x1c=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x1c)
x1a=Concatenate()([x1c,x1a])
x1a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x1a)
x1a=UpSampling2D(size=(2, 2))(x1a)

x2a=MaxPooling2D(pool_size=(8, 8), strides=(8,8))(inp)
x2a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x2a)
x2b=AveragePooling2D(pool_size=(8, 8), strides=(8,8))(inp)
x2b=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x2b)
x2c=Conv2D(16,kernel_size=(8,8), strides=(8,8),padding="same",activation='relu')(inp)
x2c=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x2c)
x2a=Concatenate()([x2c,x2a])
x2a=Concatenate()([x2b,x2a])
x2a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x2a)
x2a=Concatenate()([x1a,x2a])
x2a=UpSampling2D(size=(2, 2))(x2a)

x3a=MaxPooling2D(pool_size=(4, 4), strides=(4,4))(inp)
x3b=AveragePooling2D(pool_size=(4, 4), strides=(4,4))(inp)
x3a=Concatenate()([x3b,x3a])
x3a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x3a)
x3a=Concatenate()([x2a,x3a])
x3a=UpSampling2D(size=(2, 2))(x3a)
#x3a=Concatenate()([inp,x3a])

x4a=MaxPooling2D(pool_size=(2, 2), strides=(2,2))(inp)
x4b=AveragePooling2D(pool_size=(2, 2), strides=(2,2))(inp)
x4a=Concatenate()([x4b,x4a])
x4a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x4a)
x4a=Concatenate()([x3a,x4a])
x4a=UpSampling2D(size=(2, 2))(x4a)
x4a=Concatenate()([inp,x4a])

x=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x4a)
x=Conv2D(8,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x)
x=Conv2D(4,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x)
x=Conv2D(2,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x)
#x=Multiply()([x,negInp])
#x=add([x,inp])
'''
x1=Conv2D(16,kernel_size=(4,4), strides=(4,4),padding="same",activation='relu')(inp)
x2=Conv2D(16,kernel_size=(8,8), strides=(8,8),padding="same",activation='relu')(inp)
x3=Conv2D(16,kernel_size=(16,16), strides=(16,16),padding="same",activation='relu')(inp)
x4=Conv2D(16,kernel_size=(32,32), strides=(32,32),padding="same",activation='relu')(inp)

x1a=MaxPooling2D(pool_size=(4, 4), strides=(4,4))(inp)
x1a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x1a)
x1a=Concatenate()([x1,x1a])
x1a=UpSampling2D(size=(4, 4))(x1a)

x2a=MaxPooling2D(pool_size=(8, 8), strides=(8,8))(inp)
x2a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x2a)
x2a=Concatenate()([x2,x2a])
x2a=UpSampling2D(size=(8, 8))(x2a)

x3a=MaxPooling2D(pool_size=(16, 16), strides=(16, 16))(inp)
x3a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x3a)
x3a=Concatenate()([x3,x3a])
x3a=UpSampling2D(size=(16, 16))(x3a)

x4a=MaxPooling2D(pool_size=(32,32), strides=(32,32))(inp)
x4a=Conv2D(16,kernel_size=(3,3), strides=(1,1),padding="same",activation='relu')(x4a)
x4a=Concatenate()([x4,x4a])
x4a=UpSampling2D(size=(32,32))(x4a)


x1=Conv2D(16,kernel_size=(8,8),padding="same",activation='relu')(x1a)
x2=Conv2D(16,kernel_size=(16,16),padding="same",activation='relu')(x2a)
x3=Conv2D(16,kernel_size=(32,32),padding="same",activation='relu')(x3a)
x4=Conv2D(16,kernel_size=(64,64),padding="same",activation='relu')(x4a)
#x5=Conv2D(16,kernel_size=(128,128),padding="same",activation='relu')(x5a)

x=Concatenate()([inp,x1,x2,x3,x4,x1a,x2a,x3a,x4a])
x=Conv2D(64,kernel_size=(4,4),padding="same",activation='relu')(x)
x=Conv2D(32,kernel_size=(4,4),padding="same",activation='relu')(x)
x=Conv2D(16,kernel_size=(4,4),padding="same",activation='relu')(x)
x=Conv2D(8,kernel_size=(4,4),padding="same",activation='relu')(x)
'''

outp=x

model=Model(inp,outp)
model.summary()
model.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer = optimizers.Adam(), metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam() ,metrics=['accuracy'] )
#model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam() ,metrics=['accuracy'] )

modelname="map_network"

def lrSchedule(epoch):
    lr  = 0.8e-3
    
    #if epoch<2: lr=0.8e-2

    
    if epoch > 195:
        lr  *= 1e-4
    elif epoch > 180:
        lr  *= 1e-3
        
    elif epoch > 160:
        lr  *= 1e-2
        
    elif epoch > 140:
        lr  *= 1e-1
        
    elif epoch > 120:
        lr  *= 2e-1
    elif epoch > 60:
        lr  *= 0.5
        
    print('Learning rate: ', lr)
    
    return lr

#6.2 For autoencoder classfier
def lrSchedule2(epoch):
    lr  = 0.15e-3
    if epoch > 59:
        lr  *= 1e-3
    elif epoch > 55:
        lr  *= 1e-2
        
    elif epoch > 52:
        lr  *= 2.5e-2
        
    elif epoch > 50:
        lr  *= 5e-2
        
    elif epoch > 48:
        lr  *= 0.1
    elif epoch > 45:
        lr  *= 0.4
    elif epoch > 30:
        lr  *= 0.7
    elif epoch > 25:
        lr  *= 0.8

        
        
    print('Learning rate: ', lr)
    
    return lr

#general setting for autoencoder training model
LRScheduler     = LearningRateScheduler(lrSchedule)

filepath        = modelname+"_classifier" + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]

reso=32
'''
model.fit( samp, 
           result, 
           validation_data=(samp, result), 
           epochs=10, 
           batch_size=1,
           callbacks=callbacks_list)
'''

with open("input.csv", "r") as f:
    samp=np.loadtxt(f)
with open("output.csv", "r") as f:
    result=np.loadtxt(f)

samp=samp.reshape(int(samp.shape[0]/256/256),256,256,1)
result=result.reshape(int(result.shape[0]/256/256),256,256,1)
'''
samp-=32
result-=32
samp/=224
result/=224
'''
'''
neg=samp
neg[neg>=32]=1
neg[neg<32]=0
neg=1-neg
'''
samp/=256
result/=256

#9.3 Set the epoch to 60. As the encoding part is already been trained, it should converge faster to user defined classes.  
model.fit( samp, result,
           validation_data=(samp, result), 
           epochs=50, 
           batch_size=1,
           callbacks=callbacks_list)
  
'''sult
model.fit( [a.reshape(int(a.shape[0]),60)/reso,diffxy,diffyz,diffzx,diffxy1,diffyz1,diffzx1,diffxy2,diffyz2,diffzx2,diffxy3,diffyz3,diffzx3,diffxy4,diffyz4,diffzx4], 
           b, 
           validation_data=([a.reshape(int(a.shape[0]),60)/reso,diffxy,diffyz,diffzx,diffxy1,diffyz1,diffzx1,diffxy2,diffyz2,diffzx2,diffxy3,diffyz3,diffzx3,diffxy4,diffyz4,diffzx4], b), 
           epochs=30, 
           batch_size=2,
           callbacks=callbacks_list)
'''

'''
model.fit( [diffxy,diffyz,diffzx,diffxy1,diffyz1,diffzx1,diffxy2,diffyz2,diffzx2,diffxy3,diffyz3,diffzx3,diffxy4,diffyz4,diffzx4], 
           b, 
           validation_data=([diffxy,diffyz,diffzx,diffxy1,diffyz1,diffzx1,diffxy2,diffyz2,diffzx2,diffxy3,diffyz3,diffzx3,diffxy4,diffyz4,diffzx4], b), 
           epochs=30, 
           batch_size=2,
           callbacks=callbacks_list)
'''
model.save_weights(modelname + "_weight.hdf5")
#print( model.predict(np.expand_dims(samp[1],axis=0)) )

res_model=model.predict( np.expand_dims(samp[1],axis=0) ) 
print(samp[1].shape,res_model[0].shape,result[1].shape)


x=np.arange(256)
y=np.arange(256)

ax1 = plt.subplot(211)
ax1.set_aspect('equal')

# equivalent but more general
ax1.pcolormesh(x,y,samp[1].reshape(256,256))


ax2 = plt.subplot(212)
ax2.set_aspect('equal')
# add a subplot with no frame
ax2.pcolormesh(x,y,result[1].reshape(256,256))

ax3 = plt.subplot(222)
ax3.set_aspect('equal')
# add a subplot with no frame
ax3.pcolormesh(x,y,res_model[0][:,:,0].reshape(256,256))

plt.show()

