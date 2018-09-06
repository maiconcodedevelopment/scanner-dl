# from sklearn.externals import joblib
# from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
# import matplotlib.pyplot as plt
# import numpy
# from skimage.feature import hog
# from sklearn import svm
# from sklearn.svm import LinearSVC
# import pandas
# from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
# from sklearn.externals import joblib
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import mean_squared_error

digits = datasets.load_digits()
dataset = fetch_mldata("MNIST Original")

logic = LogisticRegression()

X_train , x_test , y_train, y_test = train_test_split(dataset.data,dataset.target,test_size=0.3,random_state=42)

# logic.fit(X_train,y_train)

print(X_train)
# r_pred = logic.predict(x_test)

# print(r_pred)
# print(numpy.sqrt(mean_squared_error(y_test,r_pred)))

# X,y = digits.data,digits.target
#
# # print(digits.images.shape)
#
# X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
#
# knn = KNeighborsClassifier(n_neighbors=7)
#
# knn.fit(X_train,y_train)
#
# score = knn.score(X_test,y_test)
#
#
# neight = numpy.arange(1,9)
# train_accuracy = numpy.empty(len(neight))
# test_accuracy = numpy.empty(len(neight))
#
# for i , k in enumerate(neight):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train,y_train)
#
#     train_accuracy[i] = knn.score(X_train,y_train)
#     test_accuracy[i] = knn.score(X_train,y_train)
#
#
# plt.title("K-NN : vars")
# plt.plot(neight,test_accuracy,label='Testing')
# plt.legend()
# plt.xlabel('Number')
# plt.ylabel('Accuracy')
# plt.show()
#
# # exit()
#
# iris = datasets.load_iris()
#
# iris_data = iris.data
# iris_target = iris.target
#
# dataframe = pandas.DataFrame(iris_data,columns=iris.feature_names)
#
# # print(dataframe.head())
#
# # print(digits.target[:-1])
# # print(digits['data'])
#
# data = numpy.array([[1,1,0],[1,1,0],[1,1,0],[1,1,1],[1,1,1],[1,1,1],[0,1,0],[0,1,0],[1,0,2],[1,0,2]])
# marc = numpy.array([1,1,1,0,0,0,2,2,3,3])
#
# model = MultinomialNB()
# model.fit(data,marc)
# predict = model.predict(numpy.array([[1,0,2]]))
#
# # print(predict)
#
# classifier = svm.SVC(gamma=0.4,C=100)
# x = digits.images
# y = digits.target
#
# n_samples = len(digits.images)
# x = x.reshape((n_samples,-1))
# classifier.fit(x[:-2],y[:-2])
#
# # print(x[-1])
#
# # print("prediction",classifier.predict(x[-2]))
# # print(classifier)
#
# # x = x.reshape(1,-1)
# # y = x.reshape(-1,1)
# #
# # exit()
# # print(x)
# # print(y)
#
# classifier.fit(x,y)
#
# modeldigits = MultinomialNB()
# # modeldigits.fit()
#
# # exit()
# # plt.figure(1,figsize=(3,3))
# # plt.imshow(digits.images[0],cmap=plt.cm.gray_r,interpolation='nearest')
# # plt.show()
#
#
# x_train ,  x_test , y_train , y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)
# # print(x_train)
# # print(x_test)
# # print(y_train)
# # print(y_test)
#
# logic = LogisticRegression()
# logic.fit(x_train,y_train)
#
# predict = logic.predict(x_test[0:10])
# # print(predict)
# score = logic.score(x_test,y_test)
#
# # print(dataset.data.shape)
# # print(dataset.target.shape)
#
#
# train_img , test_img , train_lbl , test_lbl = train_test_split(dataset.data,dataset.target,test_size=1/7.0,random_state=0)
#
# features = numpy.array(dataset.data,'int16')
# labels = numpy.array(dataset.target,'int')
#
# list_host_fd = []
#
# for feature in features:
#     fd = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
#     print('------ image -------- matrix')
#     print(fd)
#     list_host_fd.append(fd)
#
# hog_features = numpy.array(list_host_fd,'float64')
#
# clf = LinearSVC()
#
# clf.fit(hog_features,labels)
#
# joblib.dump(clf,"digits_cls.pkl",compress=3)
#
#
# logisticRegr = LogisticRegression(solver='lbfgs')
# logisticRegr.fit(train_img,train_lbl)
# predict = logisticRegr.predict(test_img[0:5])
# score = logisticRegr.score(test_img,test_lbl)
#
#
# index = 0
# misclassi = []
#
# for label , predict in zip(test_lbl,predict):
#     if label != predict:
#         misclassi.append(index)
#     index +=1
#
#
# print(misclassi)
# print(predict)
# # plt.figure(figsize=(20,4))
# # for index , (image, label) in enumerate(zip(train_img[0:5],train_lbl[0:5])):
# #     plt.subplot(1,5,index + 1)
# #     plt.imshow(numpy.reshape(image,(28,28)),cmap=plt.cm.gray)
# #     plt.title("Training",fontsize=20)
#
# print('------------')
# print(score)
#
# # print(x_test[0])
#
# plt.figure()
# plt.imshow(numpy.reshape(test_img[1],(28,28)),cmap=plt.cm.gray)
# plt.title("Training Reson")
# #
# # plt.figure(figsize=(20,4))
# # for index , (image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
# #     plt.subplot(1,5,index+1)
# #     plt.imshow(numpy.reshape(image,(8,8)),cmap=plt.cm.gray)
# #     plt.title("Training",fontsize=20)
#
# #
# plt.show()
# # print(predict)
#
# # supervides lerning
#
# # automate time - consuming or expencise manual tasks
# # exemple : doctor's dignosis
#
# # make predictions about the future
# # exemple : will a customer click on an ad or not
#
# # need labeled data
# # historical data with labels
# # experiments to get labeled data
# # crowd-sourcing labeled data
#
# #  Showing the Images and the Labels (Digits Dataset)
#
# # This section is really just to show what the images and labels look like. It usually helps to visualize your data to see what you are working with.
#
# #  Splitting Data into Training and Test Sets (Digits Dataset)
#
# # We make training and test sets to make sure that after we train our classification algorithm, it is able to generalize well to new data.
#
#
# # Scikit-learn 4-Step Modeling Pattern (Digits Dataset)
#
# # Step 1. Import the model you want to use
#
# # In sklearn, all machine learning models are implemented as Python classes
#
#
#
# # Step 4. Predict the labels of new data (new images)
#
# # Uses the information the model learned during the model training process
#
# # # Returns a NumPy Array
# # # Predict for One Observation (image)
#
# # Predict for Multiple Observations (images) at Once
#
# # Measuring Model Performance (Digits Dataset)
#
# # While there are other ways of measuring model performance, we are going to keep this simple and use accuracy as our metric.
# # To do this are going to see how the model performs on the new data (test set)
#
# # accuracy is defined as:
#
# # (fraction of correct predictions): correct predictions / total number of data points
#
# # Our accuracy was 95.3%.
#
# #  Confusion Matrix (Digits Dataset)
#
# # A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. In this section, I am just showing two python packages (Seaborn and Matplotlib) for making confusion matrices.
#
#
# #linear regression in higner dimension
#
# # regression mechanics
# # single feature
# # how do we choose a and b
# # define an error function for any given line
# # choose the line that minimizes that minimizes the error function
# #
# # the loss function
# # residual
# #
# # ordinaty lest squares (OLS) minimize sum of squares of residuals
# #
# # linear regression in higher dimensions
# #
# # to fit linear regression model here
# # need to specify 3 variables
# #
# # in higher dimensions
# #
# # must specify coefficient for each feature and the variable b
# #
# # api works exactly the same way
# # pass two arrays : features , and target
#Train/test split for regression

#As you learned in Chapter 1, train and test sets are vital to ensure that your supervised learning model is able to generalize well to new data. This was true for classification models, and is equally true for linear regression models.

#In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over all features. In addition to computing the R2
#score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models. The feature array X and target variable array y have been pre-loaded for you from the DataFrame df.

#Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to train the model, as during the course of training, the model is not only trained, but also tested on all of the available data.

#In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn's cross_val_score() function uses R2

#as the metric of choice for regression. Since you are performing 5-fold cross-validation, the function will return 5 scores. Your job is to compute these 5 scores and then take their average.

#The DataFrame has been loaded as df and split into the feature/target variable arrays X and y. The modules pandas and numpy have been imported as pd and np, respectively.


  #  Import LinearRegression from sklearn.linear_model and cross_val_score from sklearn.model_selection.
 #   Create a linear regression regressor called reg.
   # Use the cross_val_score() function to perform 5-fold cross-validation on X and y.
    #Compute and print the average cross-validation score. You can use NumPy's mean() function to compute the average.


#cross valitation motivation

#model performace is dependent os way the data is split

#not representative of the model ability to generalize

#solution cross validation

#folds
#more folds more computationally expensive

#opencv_createsemples

#100x100
#50x50