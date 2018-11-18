import matplotlib.pyplot as plt

import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import sklearn
from sklearn.model_selection import cross_val_score , GridSearchCV

data_predicts = pd.read_csv("entradas-breast.csv")
data_class = pd.read_csv("saidas-breast.csv")

def createDeep(optimizer,loss,kernel_initializer,activation,neurons):

    classification = Sequential()
    classification.add(
        Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, use_bias=True, input_dim=30))
    classification.add(Dropout(0.2))
    classification.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classification.add(Dropout(0.2))
    classification.add(Dense(units=1, activation='sigmoid'))

    # adam = keras.optimizers.Adam(lr=0.001)

    classification.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    # classification.fit(validation_split=0.3)
    return classification




optimizer_sgd = keras.optimizers.SGD(lr=0.0001, decay=0.0001)
optimizer_adam = keras.optimizers.Adam(lr=0.0001,decay=0.0001,clipvalue=0.5)

classification = KerasClassifier(build_fn=createDeep)

#tuning (ajuste) dos parametros

parametros = {'batch_size':[10,30],
              'epochs':[50,100],
              'optimizer':['adam','sgd'],
              'loss':['binary_crossentropy','hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu','tanh'],
              'neurons' : [16,8]}

grid_serach = GridSearchCV(estimator=classification,param_grid=parametros,scoring='accuracy',cv=5)
grid_serach = grid_serach.fit(data_predicts,data_class)

good_params = grid_serach.best_params_
good_precisao = grid_serach.best_score_

print(good_params)
print(good_precisao)

# result = cross_val_score(estimator=classification,X=data_predicts,y=data_class,cv=10,scoring='accuracy')

# print(result);print(result.mean());print(result.std())




#Em resumo, acc é a precisão de um lote de dados de treinamento e val_acc é a precisão de um lote de dados de teste.