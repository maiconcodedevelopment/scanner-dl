import os
import sys
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
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier , OneVsOneClassifier

#scipy
from scipy import ndimage as ndi

#skimage
from skimage import feature
from skimage.transform import rotate

class FrameCamera(object):

    def __init__ (self,rat,frame):
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


class CrossMachine(object):
    
    def __init__(self,k=1):
        self.cv = k
        self.model_cross = OneVsRestClassifier(LinearSVC(random_state=0))
        self.model_one = OneVsOneClassifier(LinearSVC(random_state=0))

    def fit_and_predict_rest(self,data_t,target_t):
        scores = cross_val_score(self.model_cross,data_t,target_t,cv=self.cv)
        mean =  numpy.mean(scores)

    def fit_and_predict_one(self,data_t,target_t):
        scores = cross_val_score(self.model_one,data_t,target_t,cv=self.cv)
        mean = numpy.mean(scores)


class LerningDataCSV(object):

    def __init__(self):
        self.df = pandas.read_csv("/opt/lampp/htdocs/scanner/scanner-vue/app/lerning/fileMachine.csv")

    def pushData(self,dataframe,target):
         df = pandas.DataFrame(dataframe)
         df['target'] = target
         df.to_csv("fileMachinetest.csv",index=False)

class LerningDigits(LerningDataCSV):

    def __init__(self):
        self.df = pandas.read_csv("/opt/lampp/htdocs/scanner/scanner-vue/app/lerning/fileMachine.csv")
        self.model = LinearSVC()
        self.model_cross = CrossMachine(9)

        data = self.df[self.df.columns[0:36]]
        data = data.values
        target = self.df['target'].values

        self.v_0 = target[target == 0]
        self.v_1 = target[target == 1]
        self.v_2 = target[target == 2]
        self.v_3 = target[target == 3]
        self.v_4 = target[target == 4]
        self.v_5 = target[target == 5]
        self.v_6 = target[target == 6]
        self.v_7 = target[target == 7]
        self.v_8 = target[target == 8]
        self.v_9 = target[target == 9]

        self.model_cross.fit_and_predict_rest(data_t=data,target_t=target)
        self.model_cross.fit_and_predict_one(data_t=data,target_t=target)

        self.model.fit(data,target)

        self.clfPack = None
        self.images = []

        self.list_host_fd = []
        self.host_features = None

    def startFeatures(self):

        for feature in self.features:
            fd = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm='L2-Hys')
            self.list_host_fd.append(fd)

        self.host_features = numpy.array(self.list_host_fd,'float64')


    def startFit(self):
        if not self.host_features is None:
            self.clf.fit(self.host_features,self.labels)
            self.neighnors.fit(self.host_features,self.labels)
            
    def saveLerning(self):
        joblib.dump(self.clf,"digits_cls.pkl",compress=3)

    def loadPkl(self):
        self.clf = joblib.load("digits_cls.pkl")
        
    def train_image(self):
        x_train , y_train , x_test , y_test  = train_test_split(self.features,self.labels,test_size=0.2,random_state=42)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** invGamma) * 255
                      for i in numpy.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)



class ScannerDespachante(LerningDigits):

    def __init__(self):
        super(self.__class__,self).__init__()
        self.scannerDocuments = []
        self.imagename = None

        self.image = None
        self.image1 = None

        self.templade_match = []

        self.altura = None
        self.largura = None

    def readImage(self,image="",name=None,rotate=False):

        self.imagename = image

        self.image = cv2.imread(image)
        self.image1 = self.image

        self.templade_match = []

        if rotate:
            rotated = imutils.rotate_bound(self.image, 27)
            self.image = rotated
            self.image1 = rotated

        self.altura , self.largura = self.image.shape[:2]

        we, h = (self.image.shape[1], self.image.shape[0])

        self.rex = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))

        rangerex = cv2.inRange(self.image,numpy.array([50,50,50],dtype='uint16'),numpy.array([200,200,200],dtype='uint16'))
        # threshold = cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),120,255,cv2.THRESH_BINARY_INV)[1]
        gradX = cv2.morphologyEx(rangerex,cv2.MORPH_CLOSE,self.rex)
        
        cnts =  cv2.findContours(gradX.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        longs = numpy.array([[0,0,0,0]])

        wV , hV = (0,0)

        for (i,c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)

            if w > wV:
                longs = numpy.array([x,y,w,h])
                wV = w
            elif (w < wV):
                continue

        x,y,w,h = longs
        self.image = self.image1[y:y + h,x:x + w]

        a = self.image.shape[0]
        l = self.image.shape[1]

        altura = 3000
        largura = int(((altura / ( a / 100)) / 100) * l)

        image =  cv2.resize(self.image,(largura,altura))
        self.image = image
        self.image1 = image

        a = self.image.shape[0]
        l = self.image.shape[1]

        tY , bY = (int(altura * 0.16),int(altura * 0.206))
        lX , rX = (int(largura * 0.136),int(largura * 0.320))

        self.image = self.image[tY:bY,lX:rX]
        a = self.image.shape[0]
        l = self.image.shape[1]

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
    
    def rectangle(self):
        cv2.rectangle(self.image,(100,100),(3,40),(255,255,0),2)

    def dilate(self):
        dilation = None
        if not self.mask is None:
            kernel = numpy.ones((5,5), numpy.uint8)
            opening = cv2.morphologyEx(self.image,cv2.MORPH_DILATE,kernel)
            dilation = cv2.dilate(self.mask,kernel, iterations=1)
            self.image = dilation
        else:
            rex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
            dilation = cv2.morphologyEx(rex,cv2.MORPH_ELLIPSE,rex)
            self.mask = dilation
            

    def black(self):
        self.image[( self.g < 120 ) & (self.r < 190) & (self.b < 190)] = [0,0,0]

    def blur(self):
        self.image = cv2.bilateralFilter(self.image,9,75,75)

    def mediablur(self):
        self.image = cv2.medianBlur(self.image,5)

    def erode(self):
        self.image = cv2.erode(self.image,numpy.zeros((5,5),numpy.uint8),iterations=1)

    def inRange(self,lower,upper,mask=False):
        self.forceHSV()
        # self.mask = cv2.inRange(self.image,numpy.array(lower,dtype='uint16'),numpy.array(upper,dtype='uint16'))
        # self.dilate()
        # self.showImage()
        # self.image = self.mask
        if mask :
            self.image = cv2.bitwise_and(self.image,self.image,mask=self.mask)

    def threshold(self):
        ret , self.image = cv2.threshold(self.image,120,255,cv2.THRESH_BINARY_INV)

    def dilateImage(self,value=(1,1),iterations=1):
        dilate = numpy.ones(value,numpy.uint8)
        self.image = cv2.dilate(self.image,dilate,iterations=iterations)

    def erodeImage(self,value=(1,1),iterations=1):
        erode = numpy.ones(value,numpy.uint8)
        self.image = cv2.erode(self.image,erode,iterations=iterations)

    def removeLineNumpy(self):
        kernel_clear = numpy.ones((2,2),numpy.uint8)
        clened = cv2.erode(self.image,kernel_clear,iterations=1)

        kernel_line = numpy.ones((1,5),numpy.uint8)
        clean_lines = cv2.erode(clened,kernel_line,iterations=6)
        clean_lines = cv2.dilate(clean_lines,kernel_line,iterations=6)

        cleaded_img_without_lines = clened - clean_lines
        cleaded_img_without_lines = cv2.bitwise_not(cleaded_img_without_lines)

        self.image = cleaded_img_without_lines


    def forceHSV(self):

        img_rgb = self.image
        img_rgb = cv2.medianBlur(img_rgb,5)

        img = self.image

        img_g = img[:,:,1]

        gray_img = cv2.cvtColor(img_rgb,cv2.IMREAD_GRAYSCALE)

        self.blackandwrite()

        input_image = self.image
        input_image = cv2.GaussianBlur(input_image,(5,5),0)
        rat , input_image = cv2.threshold(input_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        edges = cv2.Canny(input_image,50,250,apertureSize=5,L2gradient=True)
        
        img = cv2.bitwise_not(input_image)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-1)
        img_th2 = th2
        th2 = img

        horizontal = numpy.copy(th2)
        vertical = numpy.copy(th2)
        rows,cols = horizontal.shape

        horizontalsize = int(cols / 30)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))
        horizontal = cv2.erode(horizontal,horizontalStructure)
        horizontal = cv2.dilate(horizontal,horizontalStructure)

        verticalsize = int(rows / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
        vertical = cv2.erode(vertical,verticalStructure)
        vertical = cv2.dilate(vertical,verticalStructure)

        vertical = cv2.bitwise_not(vertical)

        edges = cv2.adaptiveThreshold(vertical,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,-2)

        kernel = numpy.ones((2,2),dtype="uint8")
        dilate = cv2.dilate(edges,kernel)

        smooth = numpy.copy(vertical)

        smooth = cv2.blur(smooth,(4,4))
        
        (rows, cols) = numpy.where(img == 0)
        vertical[rows, cols] = smooth[rows, cols]

        linesP = cv2.HoughLinesP(img_th2, 1, numpy.pi / 180, 50, None, 50,maxLineGap=100)
        lines = cv2.HoughLines(img_th2,1,numpy.pi / 180,200)

        if lines is not None:
            for rho , theta in lines[0]:
                a = numpy.cos(theta)
                b = numpy.sin(theta)

                x0 = a * rho
                y0 = b * rho

                x1 = int(x0 + 1500*(-b))
                y1 = int(y0 + 1500*(a)) 

                x2 = int(x0 - 1500*(-b))
                y2 = int(y0 - 1500*(a))

                pts1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 *(a)))
                pts2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 *(a)))
                cv2.line(img_rgb,pts1,pts2,(255,255,255),8)
                # cv2.line(img_rgb,(x1,y1),(x2,y2),(0,0,255),2)

        

        self.image = img_rgb
        chain = cv2.split(self.image)
        self.blackandwrite()

        hist = cv2.calcHist([chain[1],chain[0]],[0,1],None,[32,32],[0,256,0,256])

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
        cl1 = clahe.apply(self.image)

        input_image = cv2.GaussianBlur(self.image,(5,5),0)

        self.image = input_image

        rat , input_image = cv2.threshold(input_image,125,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        input_image = cv2.bitwise_not(input_image)

        self.image = input_image

        get_struct = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))

        react_struct = numpy.array((
            [0,0,1,1,1,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,1,1,1,0,0]
        ),numpy.uint8)

        self.image = cv2.morphologyEx(self.image,cv2.MORPH_OPEN,react_struct)

        react_struct = numpy.array((
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,0,0,1,1,0]
        ),numpy.uint8)


        self.image = cv2.morphologyEx(self.image,cv2.MORPH_OPEN,react_struct)

        get_close = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,10))
        
        join = numpy.array((
            [0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0]
        ),numpy.uint8)

        kernel1 = numpy.array([[0,0,0,0,0,0,0],[0,0,1,1,1,0,0],[0,0,0,0,0,0,0]],numpy.uint8)
        kernel2 = numpy.array([[1,1,1,1,1,1,1],[1,1,0,0,0,1,1],[1,1,1,1,1,1,1]],numpy.uint8)

        histomoss1 = cv2.morphologyEx(self.image,cv2.MORPH_ERODE,kernel1)
        histomoss2 = cv2.morphologyEx(self.image,cv2.MORPH_ERODE,kernel2)
        histomoss = cv2.bitwise_and(histomoss1,histomoss2)
        

        histomoss_comp = cv2.bitwise_not(histomoss)
        del_isoled = cv2.bitwise_and(self.image,self.image,mask=histomoss)


        self.dilateImage(value=(1,1),iterations=1)

    def images_read(self,path = None,list_image=[]):

        if path is None:
            path = ""
        else :
            pass


        for image in list_image:
            image_hold = cv2.imread(str(path)+"/"+image,0)
            self.templade_match.append(image_hold)

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


        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
        gradX = numpy.absolute(gradX)
        (minVal, maxVal) = (numpy.min(gradX), numpy.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, reactKernel)
        thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        self.image = thresh

    def morphologyEx(self):
        
        im2, cnts , hierarchy = cv2.findContours(self.image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        myContours = []

        for cnt in reversed(cnts):

            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            if len(approx == 4):

                rectangle = cv2.boundingRect(cnt)
                myContours.append(rectangle)


        max_width = max(myContours,key=lambda r:r[0] + r[2])[0]
        max_height = min(myContours,key=lambda r:r[3])[3]

        nearest = max_height * 1.4
        myContours.sort(key=lambda r: r[1] + r[0])

        sorted_ctrs = sorted(cnts,key=lambda ctr: cv2.boundingRect(ctr))

        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

        reacts = [ cv2.boundingRect(react) for react in cnts]

        imagesreact = numpy.array([])
        predicts = []

        amountNumber = 0
        
        for rect in myContours:

            if (rect[2] >= 15 and rect[2] <= 70) and (rect[3] >= 45 and rect[3] <= 80):
                
                amountNumber += 1

                cv2.rectangle(self.image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)

                leng = int(rect[3] * 1)
                pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                roi = self.image[pt1:pt1+leng, pt2:pt2+leng]

                roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)

                hoi_hog_fd = hog(roi,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm='L2-Hys')
                predicts.append(roi)
                predict = self.model.predict(numpy.array([hoi_hog_fd], 'float64'))
                cv2.putText(self.image, str(int(predict)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

        cv2.destroyAllWindows()

        new_featues = []
        for feature in predicts:
            img = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm='L2-Hys')
            new_featues.append(img)

        new_predict = numpy.array(new_featues,'float64')

        if amountNumber == 11:
            self.scannerDocuments.append(self.model.predict(new_predict).tolist())
        else :
            self.scannerDocuments.append(["None"])

        # print()
        # self.showImage()

        
        # marc = [0,0,2,2,9,0,1,2,5,6,6]
        # df = pandas.DataFrame(new_featues)
        # df['data'] = new_featues
        # df['target'] = marc
        # print(df)
        # self.df.append(df)
        # self.df.to_csv("fileMachine1.csv")
        # self.df = pandas.concat([self.df,df],sort=True)
        # print(self.df)
        # df.to_csv('fileMachine4.csv',index=False)
        # self.pushData(dataframe=df,target=marc)

        # print("quantidade de features : {0}".format(len(new_featues)))
        
if __name__ == "__main__":

    scanners = sys.argv[1]
    scanner = ScannerDespachante()

    for scannerDocument in scanners.replace("[","").replace("]","").split(","):
        scanner.readImage(scannerDocument)
        scanner.forceHSV()
        scanner.morphologyEx()
        scanner.showImage()

    print(scanner.scannerDocuments)
