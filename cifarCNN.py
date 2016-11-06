#%matplotlib inline
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import keras

from keras.datasets import cifar10
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt 


# reading data from cifar10 npz file and getting the training data and labels and testing data and labels
cifar10 = np.load('cifar10_data.npz')
X_train = cifar10['X_train']
y_train = cifar10['y_train']
X_test = cifar10['X_test']
y_test = cifar10['y_test']

#printig the dimensions of the dataset
print "Training data:"
print "Number of examples: ", X_train.shape[0]
print "Number of channels:",X_train.shape[1] 
print "Image size:", X_train.shape[2], X_train.shape[3]
print
print "Test data:"
print "Number of examples:", X_test.shape[0]
print "Number of channels:", X_test.shape[1]
print "Image size:",X_test.shape[2], X_test.shape[3] 


# plotting some of the data input for viewing the data in the dataset
plot = []
for i in range(1,10):
    plot_image = X_train[100*i,:,:,:].transpose(1,2,0)
    for j in range(1,10):
        plot_image = np.concatenate((plot_image, X_train[100*i+j,:,:,:].transpose(1,2,0)), axis=1)
    if i==1:
        plot = plot_image
    else:
        plot = np.append(plot, plot_image, axis=0)

plt.imshow(plot)
plt.axis('off')
plt.show()




#normalizing data using its mean and standard deviation SD
print "mean before normalization:", np.mean(X_train) 
print "std before normalization:", np.std(X_train)

mean=[0,0,0]
std=[0,0,0]
newX_train = np.ones(X_train.shape)
newX_test = np.ones(X_test.shape)
for i in xrange(3):
    mean[i] = np.mean(X_train[:,i,:,:])
    std[i] = np.std(X_train[:,i,:,:])
    
for i in xrange(3):
    newX_train[:,i,:,:] = X_train[:,i,:,:] - mean[i]
    newX_train[:,i,:,:] = newX_train[:,i,:,:] / std[i]
    newX_test[:,i,:,:] = X_test[:,i,:,:] - mean[i]
    newX_test[:,i,:,:] = newX_test[:,i,:,:] / std[i]
        
    
X_train = newX_train
X_test = newX_test

print "mean after normalization:", np.mean(X_train)
print "std after normalization:", np.std(X_train)




#setting the parameters of CNN like batch size, learning rate,number of output classes,number of epochs etc

batchSize = 50                    #-- Training Batch Size
num_classes = 10                  #-- Number of classes in CIFAR-10 dataset
num_epochs = 20                   #-- Number of epochs for training   
learningRate= 0.2                 #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch


img_rows, img_cols = 32, 32       #-- input image dimensions

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)



#creating the model now. model will be designed according to the CNN architecture we want.

from keras import backend as K
K.set_image_dim_ordering('th')					    #order of image as (channels, rows , columns)


model = Sequential()                                                #-- Sequential container.

model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=( 3,img_rows, img_cols)))    #-- 6 outputs (6 filters), 5x5 convolution kernel. -- 3 input depth (RGB)
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Convolution2D(16, 5, 5))                                  #-- 16 outputs (16 filters), 5x5 convolution kernel
model.add(Activation('relu'))                                       #-- ReLU non-linearity
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Flatten())                                                #-- eshapes a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model.add(Dense(120))                                               #-- 120 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(84))                                                #-- 84 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(num_classes))                                       #-- 10 outputs fully connected layer (one for each class)
model.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems

#prints the entire model. layer sequence, and parameters and so on.
print model.summary()

#compiling and training the model for use.
sgd = SGD(lr=learningRate, decay = lr_weight_decay)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

#-- switch verbose=0 if you get error "I/O operation from closed file"
history = model.fit(X_train, Y_train, batch_size = batchSize, nb_epoch = num_epochs,verbose=1, shuffle=True, validation_data=(X_test, Y_test))
print("compile and trained!!")



#-- summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#-- summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#-- test the network
score = model.evaluate(X_test, Y_test, verbose=0)

print 'Test score:', score[0] 
print 'Test accuracy:', score[1]



