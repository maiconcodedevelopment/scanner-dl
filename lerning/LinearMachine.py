from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy

from skimage.feature import hog

d = datasets.load_digits()
MNIST = fetch_mldata("MNIST Original")

data = d.data
target = d.target

m_data = MNIST.data
m_target = MNIST.target

features16 = numpy.array(data,'int16')
m_features16 = numpy.array(m_data,'int16')

x_train , x_test , y_train , y_test = train_test_split(data,target,test_size=0.3,random_state=42)

linear = LinearRegression()

linear.fit(x_train,y_train)

predict = linear.predict(x_test[0].reshape(1,-1))

# features = []
# for feature in features16:
    # fd = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
    # features.append(image)

print(m_features16[0])

print(data[0])

m_features = []
for feature in m_features16:
    fd = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
    m_features.append(fd)


for k in range(1,30,2):
    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(x_train,y_train)

    score = model.score(x_test,y_test)
    print("% {0} ".format(score))


print(m_features[0])

#https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/
#https://chrisalbon.com/machine_learning/basics/loading_scikit-learns_digits-dataset/

# print(features[0])
# print(features[0])
# print(len(features[0]))

# print(m_features16[0])
# print(len(m_features[0]))

print(predict)