from keras.datasets import mnist
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold


(trainX,trainY),(tesX,testY)=mnist.load_data()
trainX=trainX.reshape(trainX.shape[0],1,28,28)
tesX=tesX.reshape(tesX.shape[0],1,28,28)
trainY=to_categorical(trainY)
testY=to_categorical(testY)
trainX= trainX.astype('float32')
tesX= tesX.astype('float32')
trainY = trainY/ 255.0
testY= testY/ 255.0

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(1,28,28),data_format='channels_first'))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Flatten())
model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=3)
model.save('final_mnist.h5')

