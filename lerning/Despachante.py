import os
import cv2
import csv
import numpy
import pickle

#imutils
import matplotlib
from imutils.perspective import four_point_transform
import imutils
from imutils import contours
import pytesseract
from PIL import Image
import skimage


import matplotlib.pyplot as plt
from scipy import ndimage as ndi
#skimag
from skimage.feature import hog

#pandas
import pandas

#skleran
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
#scipy
from scipy import ndimage as ndi

#skimage
from skimage import feature
from skimage.transform import rotate

class FrameCamera(object):

    def __init__ (self,rat,frame):
        print(frame)
        self.frame = frame
        self.mask = ""


    def dilate(self):
        n = numpy.zeros((5,5),dtype=numpy.uint8)
        dilation = cv2.dilate(self.frame,n,iterations=1)
        self.mask = dilation
    
    def inRange(self,lower,upper,mask=False):
        self.mask = cv2.inRange(self.frame,numpy.array(lower,dtype='uint16'),numpy.array(upper,dtype='uint16'))
        self.frame = self.mask
    
    def cameraShow(self):
        cv2.imshow('frame',self.frame)

class LerningDigits(object):

    def __init__(self):
        self.df = pandas.read_csv("fileMachine.csv")
        self.model = LinearSVC()

        data = self.df[self.df.columns[0:36]]
        data = data.values
        target = self.df['target'].values

        print(target[target == 0])
        print(target[target == 1])
        print(target[target == 2])
        print(target[target == 3])
        print(target[target == 4])
        print(target[target == 5])
        print(target[target == 6])
        print(target[target == 7])
        print(target[target == 8])
        print(target[target == 9])

        self.model.fit(data,target)
        print(data)
        # exit()
        # print(numpy.array(self.df['data']))

        # exit()
        # self.model.fit(self.df['data'],self.df['target'])

        self.clf = LinearSVC()
        self.neighnors = KNeighborsClassifier(n_neighbors=7)

        self.clfPack = None
        self.images = []

        self.datasets = fetch_mldata("MNIST Original")

        self.features = numpy.array(self.datasets.data,'int16')
        self.labels = numpy.array(self.datasets.target,'int')

        print(self.features[0])
        print("quantaty : {0}".format(len(self.features[0])))

        self.list_host_fd = []
        self.host_features = None

    def startFeatures(self):

        for feature in self.features:
            fd = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
            self.list_host_fd.append(fd)

        print('-------- image list host --------')
        print(self.list_host_fd[0])
        self.host_features = numpy.array(self.list_host_fd,'float64')
        print(self.host_features[0])
        print("quantaty : {0}".format(len(self.host_features[0])))

    def startFit(self):
        if not self.host_features is None:
            self.clf.fit(self.host_features,self.labels)
            self.neighnors.fit(self.host_features,self.labels)
            
    def saveLerning(self):
        joblib.dump(self.clf,"digits_cls.pkl",compress=3)

    def loadPkl(self):
        self.clf = joblib.load("digits_cls.pkl")
        # self.clfPack = joblib.load("features.pkl")
        
    def train_image(self):
        x_train , y_train , x_test , y_test  = train_test_split(self.features,self.labels,test_size=0.2,random_state=42)

        print(x_train)
        print(y_train)



        # return super(LerningDigits, self).__init__()

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** invGamma) * 255
                      for i in numpy.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



class ScannerDespachante(LerningDigits):

    def __init__(self,image="",name="imageDespachange.jpg",rotate=False):
        super(self.__class__,self).__init__()

        self.startFeatures()
        self.startFit()

        self.image = cv2.imread(image)
        self.image1 = self.image

        

        if rotate:
            rotated = imutils.rotate_bound(self.image, 27)
            self.image = rotated
            self.image1 = rotated

        # rotated = imutils.rotate_bound(self.image,numpy.arange(0,160,15))
        # self.image = rotated
        # self.showImage()
        # exit()

        self.altura , self.largura = self.image.shape[:2]
        # pont = (largura / 2 , altura / 2)
        # rotation = cv2.getRotationMatrix2D(pont,25,1.0)
        # totacionado = cv2.warpAffine(self.image,rotation,(largura,altura))
        # self.image = totacionado
        # self.showImage()
        # invert = cv2.flip(self.image,1)
        # self.image = invert
        # self.showImage()

        we , h = (self.image.shape[1] , self.image.shape[0])

        self.rex = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))

        rangerex = cv2.inRange(self.image,numpy.array([50,50,50],dtype='uint16'),numpy.array([200,200,200],dtype='uint16'))
        # threshold = cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),120,255,cv2.THRESH_BINARY_INV)[1]
        gradX = cv2.morphologyEx(rangerex,cv2.MORPH_CLOSE,self.rex)
        
        cnts =  cv2.findContours(gradX.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        longs = numpy.array([[0,0,0,0]])

        # longs = numpy.concatenate( (numpy.array([[1,3]]),numpy.array([[2,3]])))
        # longs = numpy.append(longs,[[2,3,32,3]],axis=1)

        # print(longs)
        # exit()

        wV , hV = (0,0)

        for (i,c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)

            print("{0} {1}".format(w,h))
            # print(longs)


            if w > wV:
                longs = numpy.array([x,y,w,h])
                wV = w
            elif (w < wV):
                continue
                
            # longs = numpy.concatenate(( numpy.array( [[x,y,w,h]] ) ))

            # print("x : {2} , y : {3} , w {0} h {1} ".format(w,h,x,y))
            
            # if (w >= we) or (h > 1000):
                # longs.append((x,y,w,h))
                # print("W {0} W{1}".format(x,y))
                # self.image = self.image1[y:y + h,x:x + w]

        # print(a)
        # print(l)
        # self.showImage()
        # print(numpy.max(longs[:,3],axis=0))
        x,y,w,h = longs
        self.image = self.image1[y:y + h,x:x + w]
        print("x : {2} , y : {3} , w {0} h {1} ".format(w,h,x,y))
        # print(longs)

        # self.image =  self.image1[y:y + h,x:x + w]
        # print()
        # print(longs[:,2].max())

        a = self.image.shape[0]
        l = self.image.shape[1]

        # exit()

        print(" langs {0}".format(longs))

        print("a: {0} , l : {1}".format(a,l))

        altura = 3000
        largura = int(((altura / ( a / 100)) / 100) * l)

        # print('_____')
        print("altura : {0} , largura : {1}".format(altura,largura))

        # print("{:10.2f}".format(altura))

        image =  cv2.resize(self.image,(largura,altura))
        self.image = image
        self.image1 = image
        # self.name = name
        a = self.image.shape[0]
        l = self.image.shape[1]

        print(a)
        print(l)

        tY , bY = (int(altura * 0.16),int(altura * 0.206)) 
        lX , rX = (int(largura * 0.136),int(largura * 0.312))

        print("top: {0} bottom: {1} , left : {2} right  : {3}".format(tY,bY,lX,rX))

        # print(a)
        # print(l)
        # m = int((a / 2))
        # self.image = self.image[:m,:]

        # exit()
        # print(rex)
        # print(longs)
        # self.image = self.image
        # self.showImage()
        # print(self.image.shape)

        #calc record


        #cortar a imagem
        self.image = self.image[tY:bY,lX:rX]
        a = self.image.shape[0]
        l = self.image.shape[1]

        # print(a)
        # print(l)

        altura = 200
        largura = int(((altura / ( a / 100)) / 100) * l)

        self.image = cv2.resize(self.image,(largura,altura))
        self.image1 = self.image

        self.b = self.image[...,0]
        self.g = self.image[...,1]
        self.r = self.image[...,2]
        
        self.hstack = None
        self.mask = None

    def blackandwrite(self):
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

    def brightness(self,gamma=1.0):
        adjusted = adjust_gamma(self.image, gamma=gamma)
        # cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        self.image = adjusted

    def constrast(self,value,bright):
        img = numpy.int16(self.image)
        img = img * (value / 127 + 1) - value + bright
        img = numpy.clip(img, 0, 255)
        img = numpy.clip(img, 0, 255)
        self.image = numpy.uint8(img)

    def save(self,name):
        cv2.imwrite(name,self.image)

    def showImage(self):
        cv2.namedWindow('imagem',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('imagem',self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def green(self):
        b, g, r = self.image[..., 0], self.image[..., 1], self.image[..., 2]
        self.image[((b > g) & (b > r) & ((b - r) > 30))] = [255, 255, 255]
        # self.image[( ( self.g > self.b ) &( self.g > self.r ) & (self.g > 140) )] = [255,255,255]

    # def threshold(self):
    #     ret , self.image = cv2.threshold(self.image,127,255,cv2.THRESH_BINARY)
    
    def rectangle(self):
        cv2.rectangle(self.image,(100,100),(3,40),(255,255,0),2)

    def dilate(self):
        kernel = numpy.ones((10,10), numpy.uint8)
        dilation = None

        if not self.mask is None:
            rex = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
            dilation = cv2.dilate(self.mask, kernel, iterations=1)
            # opening = cv2.morphologyEx(self.mask,cv2.MORPH_GRADIENT,kernel)
            dilation = cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,rex)
            self.mask = dilation
        else:
            opening = cv2.morphologyEx(self.image,cv2.MORPH_DILATE,kernel)
            dilation = cv2.dilate(self.image,kernel, iterations=1)
            self.image = dilation

    def black(self):
        self.image[( self.g < 120 ) & (self.r < 190) & (self.b < 190)] = [0,0,0]

    def blur(self):
        self.image = cv2.bilateralFilter(self.image,9,75,75)

    def mediablur(self):
        self.image = cv2.medianBlur(self.image,5)

    def erode(self):
        self.image = cv2.erode(self.image,numpy.zeros((5,5),numpy.uint8),iterations=1)

    def inRange(self,lower,upper,mask=False):
        self.mask = cv2.inRange(self.image,numpy.array(lower,dtype='uint16'),numpy.array(upper,dtype='uint16'))
        self.dilate()
        self.image = self.mask
        # self.showImage()
        if mask :
            self.image = cv2.bitwise_and(self.image,self.image,mask=self.mask)

        # if mask :
    def threshold(self):
        ret , self.image = cv2.threshold(self.image,120,255,cv2.THRESH_BINARY_INV)

        
    def findContours(self):
        
        ret , self.image = cv2.threshold(self.image,0,255,cv2.THRESH_BINARY)
        refCnts = cv2.findContours(self.image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
        refCnts = contours.sort_contours(refCnts,method='left-to-right')[0]

        digits = {}

        for (i ,c ) in enumerate(refCnts):

            (x , y , w, h ) = cv2.boundingRect(c)

            roi = self.image[y:y + h ,x:x + w]
            cv2.imshow('image{0}'.format(i),roi)

            digits[i] = roi

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # for c in cnts:-se
        #     print(c)

        #     peri = cv2.arcLength(c,True)
        #     approx = cv2.approxPolyDP(c,0.02 * peri,True)

        #     if len(approx) == 4:
        #         displayCnt = approx
        #         break
        
        # warped = four_point_transform(gray,displayCnt.reshape(4,2))
        # output = four_point_transform(self.image,displayCnt.reshape(4,2))

        # self.image = warped
        # (cnts , _) = cv2.findContours()

    def maskRecord(self):
        self.image[numpy.where((self.image == [0]).all(axis = 1))] = [255]


    def graussianBlur(self):
        self.image = cv2.GaussianBlur(self.image,(11,11),0)

    
    def react(self):
        reactKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        image = imutils.resize(self.image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,reactKernel)

        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	    ksize=-1)
        gradX = numpy.absolute(gradX)
        (minVal, maxVal) = (numpy.min(gradX), numpy.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, reactKernel)
        thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
         
        # apply a second closing operation to the binary image, again
        # to help close gaps between credit card number regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        self.image = thresh

    def morphologyEx(self):

        gray = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)

        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        reactKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))

        gradX = cv2.morphologyEx(self.image,cv2.MORPH_CLOSE,sqKernel)

        rat ,thresh = cv2.threshold(gradX,90,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        morplot = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)

        # self.save('teste1.png')/

        cnts = cv2.findContours(morplot.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        reacts = [ cv2.boundingRect(react) for react in cnts ]
        
        # print(reacts)
        rex = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))

        imagesreact = numpy.array([])
        predicts = []

        train = KNeighborsClassifier(n_neighbors=2)

        #bunch
        for rect in reacts:
            cv2.rectangle(self.image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3) 
            leng = int(rect[3] * 1)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = thresh[pt1:pt1+leng, pt2:pt2+leng]

            # if (rect[2] >= 10 and rect[2] <= 80) and (rect[3] >= 30 and rect[3] <= 90):
            
            # print("react 3 {0}".format(rect[3]))

            roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
            # roi = cv2.dilate(roi,(-2,-2))

            # print('---------------------')
            # print(roi)
            # print('---------------------')
            # dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,rex)

            # cv2.imshow('image',roi)
            # cv2.waitKey(0)

            hoi_hog_fd = hog(roi,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
            
            # print('--------------------- hoh')
            # print(hoi_hog_fd)
            # print('--------------------- hoh')

            if (rect[3] >= 50):
                self.images.append(roi)
                predict = self.model.predict(numpy.array([hoi_hog_fd], 'float64'))
                print(predict)
                cv2.putText(self.image, str(int(predict[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

            # cv2.imshow('image',roi)
            # nbr = self.clf.predict(numpy.array([hoi_hog_fd],'float64'))
            # predicts.append(nbr)
        # cv2.imshow("machine",gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.showImage()
        self.image = thresh
        # model_s = LinearSVC()
        # model_s.fit(self.images,numpy.array([0,0,6,9,4,9,3,8,1,9,0]))
        new_featues = []
        for feature in self.images:
            img = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
            new_featues.append(img)

        # print(new_featues)
        new_featues = numpy.array(new_featues,'float64')
        # marc = [0,1,1,1,2,7,2,9,7,0,1]
        # df = pandas.DataFrame(new_featues)
        # df['data'] = new_featues
        # df['target'] = marc
        # self.df.append(df)
        # self.df.to_csv("fileMachine1.csv")
        # self.df = pandas.concat([self.df,df],sort=True)
        # print(self.df)
        # df.to_csv('fileMachine1.csv',index=False)

        # print("quantidade de features : {0}".format(len(new_featues)))

        # with open('cvfile.csv','w') as wireFile:
            # whiter = csv.writer(wireFile)
            # for (index,row) in enumerate(new_featues,start=0):
                # whiter.writerow(zip(row,marc[index]))

        # wireFile.close()

        exit()

        print(new_featues)

        quantaty_fe = len(new_featues[0])

        model = LinearSVC()
        model.fit(new_featues,numpy.array([0,1,1,1,2,7,2,9,7,0,1]))

        # joblib.dump(new_featues,'digits_features.pkl',compress=3)
        # print(model.target)

        print('----------- 2')

        for rect in reacts:
            cv2.rectangle(
                self.image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
            leng = int(rect[3] * 1)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = thresh[pt1:pt1+leng, pt2:pt2+leng]

            # if (rect[2] >= 10 and rect[2] <= 80) and (rect[3] >= 30 and rect[3] <= 90):

            # print("react 3 {0}".format(rect[3]))

            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            # roi = cv2.dilate(roi,(-2,-2))

            # print('---------------------')
            # print(roi)
            # print('---------------------')
            # dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,rex)

            # cv2.imshow('image', roi)
            # cv2.waitKey(0)

            hoi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
                14, 14), cells_per_block=(1, 1), visualise=False)

            # print('--------------------- hoh')
            # print(hoi_hog_fd)
            # print('--------------------- hoh')

            if (rect[3] >= 50):
                if quantaty_fe == len(hoi_hog_fd):
                    pre = model.predict(numpy.array([hoi_hog_fd],'float64'))
                    print(pre)

        # model.fit(new_featues,numpy.array([0,0,0]))

        exit()
        print('---------- numpy append')
        print(imagesreact)
        print('-------------- model -------------------')
        print(len(self.images))
        print(self.images[0])
        print("quantity : {0}".format(len(self.images[0])))

        images = numpy.array(self.images[0],'int16')
        print(image[0])
        new_data = [] 

        for feature in  images:
            img = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
            new_data.append(img)

        print(new_data)
        exit()
        # new_data_float64 = numpy.array(new_data,'float64')

        print(new_data[0])
        print('--------------- <> -----------')
        print(new_data_float64[0])

        print('-------------- model -------------------')
        self.showImage()
        # cv2.destroyAllWindows()
        exit()

        locs = []

        for (i ,c )in enumerate(cnts):

            (x,y,w,h) = cv2.boundingRect(c)
            ar = w / float(h)

            print("{0} {1}".format(w,h))
            if ( w >= 10 and w <= 80) and (h >= 30 and h <= 90):
                locs.append((x,y,w,h))


        # print(locs)
        # exit()
        locs = sorted(locs,key=lambda x:x[0])
        output = []

        for (i ,(x,y,w,h)) in enumerate(locs):
            groupOutput = []

            group = gray[y:y + h,x:x + w]
            cv2.imshow("images",group)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # digitCnts = cv2.findContours(group,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
            # digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

            # for c in digitCnts:
            #     print(c)

        
        self.showImage()
        self.image = thresh

if __name__ == "__main__" :
    print('------- start machine lerning --------')

    # lerning = LerningDigits()
    # lerning.startFeatures()
    # lerning.fitClf()
    # lerning.saveLerning()

    # video = cv2.VideoCapture(0)

    # while (True):
    #     (grabbed , frame) = video.read()

    #     horl = FrameCamera(rat=grabbed,frame=frame)
    #     horl.inRange([0,0,0],[100,100,100],mask=True)
    #     horl.cameraShow()

    #     if cv2.waitKey(1) == 27:
    #         break

    # video.release()
    # cv2.destroyAllWindows()

    # digits = datasets.load_digits()

    # im = numpy.zeros((128,128))

    # im[32:-32,32:-32] = 1

    # edges = feature.canny(im)


    # fig , (x1,x2,x3) = plt.subplots(nrows=1,ncols=3,figsize=(8,2),sharex=True,sharey=True)

    # x2.imshow(edges,cmap=plt.cm.gray)
    # x2.axis('off')
    # x2.set_title('canny',fontsize=20)

    # fig.tight_layout()

    # plt.show()

    # print(digits)

    img = ScannerDespachante(image='images/Scanner_20180822 (3).png')
    # img.findContours()
    # img.findContours()
    # img.black()
    # img.react()
    img.inRange([0,0,0],[110,110,110],mask=False)
    # img.green()
    # img.black()
    # img.blackandwrite()
    # img.threshold()
    # img.dilate()
    img.morphologyEx()
    # img.dilate()
    img.showImage()
    # data = pandas.read_csv("cvfile.csv",names=['data','target'])
    # print(data[:,data.columns])
    # df = pandas.DataFrame([1,2,32],columns=['data'])
    # print(df)
    # df['target'] = marc
    
    # df.to_csv('fileMachine.csv',index=False) 
    
