from keras.datasets import cifar10

(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

print(x_train.shape[0])
print(x_train[0][0].shape)
print(x_train.reshape((x_train.shape[0],64,(32 / 2),3)))