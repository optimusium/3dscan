# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:09:00 2020

@author: boonping
"""

import numpy as np
import matplotlib as plt
#import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda,Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
#from tensorflow.keras import backend

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
reso=32           

                        
with open('input_full.csv','r') as opener1:
    inpu=np.loadtxt(opener1)
    inpu=inpu.reshape(inpu.shape[0],5,3) #.astype('float')
    #print(inpu)

diff=np.diff(inpu,axis=1) #.astype('float')
#print(diff)
diff[diff==0]=0.00001

grad_xy=diff[:,:,0]/diff[:,:,1]
grad_xz=diff[:,:,0]/diff[:,:,2]
grad_yz=diff[:,:,1]/diff[:,:,2]

grad_xy[grad_xy<-reso]=-reso*2
grad_xz[grad_xz<-reso]=-reso*2
grad_yz[grad_yz<-reso]=-reso*2
grad_xy[grad_xy>reso]=reso*2
grad_xz[grad_xz>reso]=reso*2
grad_yz[grad_yz>reso]=reso*2

grad_xy/=reso*2
grad_xz/=reso*2
grad_yz/=reso*2
#print(grad_xy)
#grad_xy=grad_xy.reshape(grad_xy.shape[0],4)
#print((grad_xy))

inpu=inpu.reshape(inpu.shape[0],15)
inpu2=inpu[:,0:12]/reso

diff=diff.reshape(diff.shape[0],12)/reso


with open('output_full.csv','r') as opener1:
    outpu=np.loadtxt(opener1)
    outpu=outpu.reshape(outpu.shape[0],4*3)/reso #.astype('float')
    #print(inpu)


inp=Input(shape=(12,))
inpb=Input(shape=(12,))
inp2=Input(shape=(4,))
inp3=Input(shape=(4,))
inp4=Input(shape=(4,))
x=Dense(12,activation='linear')(inp)
xb=Dense(60,activation='linear')(inpb)
#x2=inp2
x2=Concatenate()([inp2,inp3])
x2=Concatenate()([x2,inp4])
x2=Dense(60)(x2)
xb=Concatenate()([xb,x2])

xb=Dense(128)(xb)
xb=Dense(128)(xb)
xb=Dense(60)(xb)
xb=Dense(12)(xb)
x=add([x,xb])
#x=Lambda(quantize)(x)
#x=Concatenate()([x,xb])
#x=Dense(32,activation='tanh')(x)
#x=Dense(12,activation='tanh')(x)
output=x

model=Model([inp,inpb,inp2,inp3,inp4],output)
model.summary()
model.compile(optimizer=optimizers.RMSprop(), loss='mean_squared_error',metrics=['accuracy'] )

def lrSchedule(epoch):
    lr  = 0.75e-3
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

modelname='interpNetwork'
LRScheduler     = LearningRateScheduler(lrSchedule)

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]
model.save(filepath)
model.fit([inpu2,diff,grad_xy,grad_xz,grad_yz], 
            outpu, 
            validation_data=([inpu2,diff,grad_xy,grad_xz,grad_yz], outpu), 
            epochs=3, 
            batch_size=1,
            callbacks=callbacks_list)
#model.save(filepath)
model.save_weights(modelname + "_weight.hdf5")
print(inpu[15].reshape(5,3))
print(model.predict([inpu2,diff,grad_xy,grad_xz,grad_yz])[15].reshape(4,3)*reso )


loss=model.history['loss']
val_loss=model.history['val_loss']
epochs = range(140)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
