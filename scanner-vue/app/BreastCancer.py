import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.collections as collections

from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.datasets import load_iris , load_digits , load_boston , load_files

import tensorflow
import keras

from keras.optimizers import SGD , Adamax ,Adagrad ,Adadelta
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.activations import relu , softmax , softplus , softsign , tanh , sigmoid , hard_sigmoid

from keras.callbacks import EarlyStopping

# ocorre quando muitas camadas têm encostas muito pequenas (por exemplo, devido a estar na parte plana da curva de tanh)

data_predicts = pd.read_csv("entradas-breast.csv")
data_class = pd.read_csv("saidas-breast.csv")

predicts_train , predicts_test , class_train , class_test = train_test_split(data_predicts,data_class,test_size=0.25,random_state=0)

print(class_test)

classification = Sequential()
classification.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',use_bias=True,input_dim=30))
classification.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))
classification.add(Dense(units=1,activation='sigmoid'))

optimizer_sgd = keras.optimizers.SGD(lr=0.0001,decay=0.0001)
optimizer_adam = keras.optimizers.Adam(lr=0.001,decay=0.0001,clipvalue=0.5)

# adam = keras.optimizers.Adam(lr=0.001)

classification.compile(optimizer=optimizer_adam,loss='binary_crossentropy',metrics=['binary_accuracy'])
history = classification.fit(predicts_train,class_train,batch_size=10,epochs=50,verbose=False)

plt.plot(history.history['loss'],'r',history.history['binary_accuracy'],'b')
plt.xlabel('Epochs')
plt.ylabel('Loss Function')
plt.show()

predicts = classification.predict(predicts_test)
predicts = (predicts > 0.5)

predicts_score = accuracy_score(class_test,predicts)
matrix = confusion_matrix(class_test,predicts)
result = classification.evaluate(predicts_test,class_test)

weights_1 = classification.layers[0].get_weights()
weights_2 = classification.layers[1].get_weights()
weights_3 = classification.layers[2].get_weights()

print(weights_1)
print(weights_2)
print(weights_3)

print(predicts_score);print(matrix);print(result)

classification.summary()

exit()

titles = ["Austrain","BlockJain","People","Points"]
datas = [341,45,544,1234]

data = {'titles' : titles , 'data' : datas}

df = pd.DataFrame(data)

df.i

print(df)

# digits = load_digits()
# input_datas = digits.data
# targets = digits.target
#
# n_cols = targets.shape[0]
#
# model = Sequential()
# model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(1))


predicts = pd.read_csv("entradas-breast.csv")
outputs = pd.read_csv("saidas-breast.csv")

predicts_train , predicts_test , class_train , class_test = train_test_split(predicts,outputs,test_size=0.15)

def createDeep():
    classification = Sequential()
    classification.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform',input_dim=30))
    classification.add(Dense(units=16,activation='relu',kernel_initializer='random_uniform'))
    classification.add(Dense(units=1,activation='sigmoid'))

    optimazer = keras.optimizers.Adam(lr=0.001,decay=0.0001,clipvalue=0.5)
    optimazer1 = keras.optimizers.Adamax(lr=0.001,decay=0.001,chipvalue=0.5)

    classification.compile(optimizer=optimazer,loss='binary_crossentropy',metrics=['binary_accuracy'])

    return classification

class_keras = KerasClassifier(build_fn=createDeep,epochs=100,batch_size=10)
result = cross_val_score(estimator=class_keras,X=predicts,y=outputs,cv=10,scoring='accuracy')
mean = result.mean()
desvio = result.std()

print(result)
print(mean)
print(desvio)

# classification.fit(predicts_train,class_train,batch_size=10,epochs=100)
#
# weight_0 = classification.layers[0].get_weights()
# weight_1 = classification.layers[1].get_weights()
# weight_2 = classification.layers[2].get_weights()
#
# predicts_classifier = classification.predict(predicts_test)
# predicts_classifier = (predicts_classifier > 0.5)
#
# print(predicts_classifier)
#
# precis = accuracy_score(class_test,predicts_classifier)
# matrix = confusion_matrix(class_test,predicts_classifier)
#
# result = classification.evaluate(predicts_test,class_test)
#
# print(precis)
# print(matrix)
# print(result)