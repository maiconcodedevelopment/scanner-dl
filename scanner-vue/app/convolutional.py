import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense ,Dropout ,Conv2D , Flatten , MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

seed = 5
np.random.seed(seed)

(x,y),(x_t,y_t) = mnist.load_data()

print(x[0].reshape(28,28,1))

predicts_weppers = x.reshape(x.shape[0],28,28,1).astype('float32') / 255
prdicts_class = np_utils.to_categorical(y,10)

lfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
result = []

a , b = (np.zeros(5),np.zeros(shape=(prdicts_class.shape[0],1)))

for indice_train , index_test in lfold.split(predicts_weppers,np.zeros(shape=(prdicts_class.shape[0],1))):
    print("index train: {0} , index test : {1}".format(indice_train,index_test))

    md = Sequential()
    md.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu',dilation_rate=2))
    md.add(MaxPooling2D(pool_size=(2,2)))
    md.add(Flatten())
    md.add(Dense(units=128,activation='relu'))
    md.add(Dense(units=10,activation='softmax'))

    md.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    md.fit(predicts_weppers[indice_train],prdicts_class[indice_train],batch_size=128,epochs=5)
    precisao = md.evaluate(predicts_weppers[index_test],prdicts_class[index_test])
    result.append(precisao[1])


print(result)
print(sum(result) / len(result))

exit()

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

predicts_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32') / 255
predicts_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32') / 255

class_train = np_utils.to_categorical(y_train,10)
class_test = np_utils.to_categorical(y_test,10)

classification = Sequential()

classification.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
classification.add(BatchNormalization())
classification.add(MaxPooling2D(pool_size=(2,2)))
# classification.add(Flatten())

classification.add(Conv2D(32,(3,3),activation='relu'))
classification.add(BatchNormalization())
classification.add(MaxPooling2D(pool_size=(2,2)))
classification.add(Flatten())

classification.add(Dense(units=128,activation='relu'))
classification.add(Dropout(0.2))
classification.add(Dense(units=128,activation='relu'))
classification.add(Dropout(0.2))
classification.add(Dense(units=10,activation='softmax'))

classification.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
classification.fit(predicts_train,class_train,batch_size=128,epochs=5,validation_data=(predicts_test,class_test))

result = classification.evaluate(predicts_test,class_test)

gerador_train = ImageDataGenerator(rotation_range=7,horizontal_flip=True,shear_range=0.2,height_shift_range=0.07,zoom_range=0.2)
gerador_test = ImageDataGenerator()

base_train = gerador_train.flow(predicts_train,predicts_test,batch_size=128)
base_test = gerador_test.flow(class_train,class_test,batch_size=128)

classification.fit_generator(base_train,steps_per_epoch=60000 / 128,validation_data=base_test,epochs=5,validation_steps=10000 / 128)
classification.layers[0].get_weights()


print(result)



print(class_test)
print(class_test)

exit()




labels = ["shoe","shirt","shoe","shirt","dress","dress"]

n_categories = 3

categotiries = np.array(["shirt","dress","shoe"])

ohe_labels = np.zeros((len(labels),n_categories))


for ii in range(len(labels)):
    jj = np.where(categotiries == labels[ii])
    print(jj)
    ohe_labels[ii,jj] = 1
    print(ohe_labels)

#test_labels * predicts next -> test_labels * predicts_next / len(predicts)

array = np.array([0,0,0,0,0,1,1,1,1,1])
kernel = np.array([-1,1])
conv = np.array([0,0,0,0,0,0,0,0])

conv[0] = (kernel * array[0:2]).sum()
conv[1] = (kernel * array[1:3]).sum()
conv[2] = (kernel * array[2:4]).sum()

print(conv)

for ii in range(8):
    print((kernel * array[ii:ii+2]))
    conv[ii] = (kernel * array[ii:ii+2]).sum()

print(conv)


#kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
# result = np.zeros(im.shape)

# Output array
# for ii in range(im.shape[0] - 3):
    # for jj in range(im.shape[1] - 3):
        # result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
# print(result)