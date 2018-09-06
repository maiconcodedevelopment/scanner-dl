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


class LerningDataCSV(object):

    def __init__(self):
        self.df = pandas.read_csv("fileMachine.csv")

    def pushData(self,dataframe,target):
         df = pandas.DataFrame(dataframe)
         df['target'] = target
         df.to_csv("fileMachinetest.csv",index=False)

class LerningDigits(LerningDataCSV):

    def __init__(self):
        self.df = pandas.read_csv("fileMachine.csv")
        self.model = LinearSVC()

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

        # print(target[target == 0])
        # print(target[target == 1]) 
        # print(target[target == 2])
        # print(target[target == 3])
        # print(target[target == 4])
        # print(target[target == 5])
        # print(target[target == 6])
        # print(target[target == 7])
        # print(target[target == 8])
        # print(target[target == 9])

        self.model.fit(data,target)

        print(target)

        # print(target)
        
        x_train , y_train , x_test , y_test = train_test_split(data,target,test_size=1/7.0,random_state=0)

        model = LinearSVC()
        model.fit(data,target)

        predict = model.predict(numpy.array(data[2:4]))

        # print(self.df)
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
        self.imagename = image

        self.image = cv2.imread(image)
        self.image1 = self.image

        self.templade_match = []        

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
        self.showImage()

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
        self.showImage()
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
        dilation = None
        if not self.mask is None:
            kernel = numpy.ones((5,5), numpy.uint8)
            opening = cv2.morphologyEx(self.image,cv2.MORPH_DILATE,kernel)
            dilation = cv2.dilate(self.mask,kernel, iterations=1)
            self.image = dilation
        else:
            rex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
            # dilation = cv2.dilate(self.mask, kernel, iterations=1)
            # opening = cv2.morphologyEx(self.mask,cv2.MORPH_GRADIENT,kernel)
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

        # if mask :
    def threshold(self):
        ret , self.image = cv2.threshold(self.image,120,255,cv2.THRESH_BINARY_INV)

    def forceHSV(self):
        self.showImage()

        img_rgb = self.image

        img = self.image

        self.blackandwrite()

        input_image = self.image
        input_image = cv2.GaussianBlur(input_image,(5,5),0)
        rat , input_image = cv2.threshold(input_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        edges = cv2.Canny(input_image,50,250,apertureSize=5,L2gradient=True)

        self.image = edges
        self.showImage()
        
        img = cv2.bitwise_not(input_image)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-1)
        th2 = img

        horizontal = numpy.copy(th2)
        vertical = numpy.copy(th2)
        rows,cols = horizontal.shape

        horizontalsize = int(cols / 30)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))
        horizontal = cv2.erode(horizontal,horizontalStructure)
        horizontal = cv2.dilate(horizontal,horizontalStructure)

        cv2.imshow("horizontal",horizontal)

        verticalsize = int(rows / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
        vertical = cv2.erode(vertical,verticalStructure)
        vertical = cv2.dilate(vertical,verticalStructure)

        cv2.imshow("vertical",vertical)

        vertical = cv2.bitwise_not(vertical)

        cv2.imshow("vertical-not",vertical)
        
        edges = cv2.adaptiveThreshold(vertical,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,-2)
        
        kernel = numpy.ones((2,2),dtype="uint8")
        dilate = cv2.dilate(edges,kernel)

        cv2.imshow("dilate vertical edges",dilate)

        smooth = numpy.copy(vertical)

        smooth = cv2.blur(smooth,(4,4))
        
        cv2.imshow("edges",edges)

        (rows, cols) = numpy.where(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        cv2.imshow("vertical recebi",vertical)

        linesP = cv2.HoughLinesP(img, 1, numpy.pi / 180, 50, None, 50, 10)
        lines = cv2.HoughLines(img,1,numpy.pi / 180,200)

        if lines is not None:
            print('------------------------------------- asdf asd f-------------------------------')
            for rho , theta in lines[0]:
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pts1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 *(a)))
                pts2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 *(a)))
                cv2.line(img_rgb,pts1,pts2,(255,255,255),10)
        
        self.image = img_rgb
        self.image = cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        input_image = cv2.GaussianBlur(self.image,(5,5),0)
        rat , input_image = cv2.threshold(input_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        input_image = cv2.bitwise_not(input_image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,10))
        closing = cv2.morphologyEx(input_image,cv2.MORPH_CLOSE,kernel)

        open_img = numpy.ones((5,5),numpy.uint8)
        open_img = cv2.morphologyEx(closing,cv2.MORPH_OPEN,open_img)

        self.image = open_img
        self.showImage()
        exit()
        # self.save("line_image.jpg")

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(input_image, (l[0], l[1]), (l[2], l[3]), (255,255,0), 3, cv2.LINE_AA)
        
        # vertical[rows,cols] = smooth[rows,cols]
        self.image = input_image
        self.showImage()
        
        exit()
        # kernel = numpy.ones((5,5),numpy.float32) / 25
        # dst = cv2.filter2D(self.image,-1,kernel)
        # self.image = dst

        

        imgrgb = self.image
        
        # low = numpy.array([190,190,190])
        # high = numpy.array([255,255,255])

        # image = cv2.cvtColor(self.image,cv2.COLOR_HSV2RGB)

        # image_mask = cv2.inRange(image,low,high)

        gray = self.image

        # (thresh,img_bin) = cv2.threshold(self.image,165,255,cv2.THRESH_BINARY)

        # img = self.image
        # img_bin = 255 - img

        # verticle = numpy.array(img).shape[1]
        # verticle1 = numpy.array(img).shape[1] // 40

        # verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        # hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

        # img_temp = cv2.erode(img_bin,verticle_kernel,iterations=3)
        # self.image = img_temp
        # self.showImage()

        # vertible_lines_img = cv2.dilate(img_temp,verticle_kernel,iterations=3)

        # self.image = vertible_lines_img
        # self.showImage()

        # print(hori_kernel)
        # print(len(verticle_kernel))
        # print(verticle_kernel)

        # print(verticle)
        # print(verticle1)
        
        # exit()
        
        
        # self.save("imagecinza.png")


        input_image = self.image

        th3 = cv2.adaptiveThreshold(input_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,1)
        # exit()
        input_image = cv2.GaussianBlur(input_image,(5,5),0)

        rat , input_image = cv2.threshold(input_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.image = input_image
        self.showImage()

        exit()
        edges = cv2.Canny(self.image,50,100,apertureSize=3)
        minLineLength = 50
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=numpy.pi/180, threshold=100,lines=numpy.array([]), minLineLength=minLineLength,maxLineGap=500)

        a,b,c = lines.shape
        for i in range(a):
            x = lines[i][0][0] - lines [i][0][2]
            y = lines[i][0][1] - lines [i][0][3]
            if x!= 0:
                if abs(y/x) <1:
                    cv2.line(self.image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 1, cv2.LINE_AA)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
        cv2.imwrite('houghlines.jpg', gray)
        cv2.imshow('img', gray)
        cv2.waitKey(0)

        self.showImage()
        exit()
        
        # lines = cv2.HoughLines(image=edges)
        

        self.showImage()

        exit()

        input_image = cv2.bitwise_not(input_image)

        self.image = input_image
        
        self.showImage()

        exit()

        self.image = cv2.morphologyEx(input_image,cv2.MORPH_CLOSE,numpy.ones((1,1),numpy.uint8))

        module1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(100,5))
        module2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

        histormiss1 = cv2.morphologyEx(input_image,cv2.MORPH_ERODE,module1)
        histormiss2 = cv2.morphologyEx(input_image,cv2.MORPH_ERODE,module2)
        histormiss3 = cv2.bitwise_and(histormiss1,histormiss2)

        image = cv2.bitwise_and(input_image,input_image,mask=histormiss3)


        # laplacian = cv2.Laplacian(histormiss3,cv2.CV_64F)
        # sobelx = cv2.Sobel(self.image,cv2.CV_64F,1,0,ksize=2)
        # sobely = cv2.Sobel(self.image,cv2.CV_64F,0,1,ksize=2)

        cv2.imshow("sobelx",image)
        cv2.waitKey(0)
        
        # cv2.imshow("sobely",sobely)
        # cv2.waitKey(0)
        
        cv2.destroyAllWindows()

        self.image = histormiss3
        self.showImage()
        exit()

        thresh = cv2.adaptiveThreshold(histormiss3,255,cv2.CALIB_CB_ADAPTIVE_THRESH,cv2.THRESH_BINARY_INV,5,1)

        self.image = thresh
        self.showImage()

        # exit()

        im2 , contours , hierarchy = cv2.findContours(histormiss3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours,key=cv2.contourArea,reverse=True)

        cvts = [cv2.boundingRect(cvt) for cvt in contours]

        i = 0

        template = cv2.imread('robot/number-2.jpg',0)
        w, h = template.shape[::-1]

        rest = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        thresholdte = 0.8
        loc = numpy.where(rest >= thresholdte)

        if len(self.templade_match) > 0:

            for image in self.templade_match:

                w , h = image.shape[::-1]

                rest = cv2.matchTemplate(gray,image,cv2.TM_CCOEFF_NORMED)
                loc = numpy.where(rest >= thresholdte)

                for pt in zip(*loc[::-1]):
                    cv2.rectangle(self.image,pt,(pt[0] + w,pt[1] + h),(255,0,0),2)

                # for rect in cvts:

                #      if (rect[2] >= 30 and rect[2] <= 80) and (rect[3] >= 30 and rect[3] <= 90):
                #          leng = int(rect[3] * 1)
                #          pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                #          pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                #          roi = imgrgb[pt1:pt1+leng, pt2:pt2+leng]

                #          cv2.imshow("image",roi)
                #          cv2.waitKey(0)
                #          cv2.imwrite("number-"+str(i)+".jpg",roi)

                #          i += 1
                #          cv2.rectangle(self.image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)

        # print(loc)
        cv2.destroyAllWindows()
        self.showImage()
        exit()

    def images_read(self,path = None,list_image=[]):

        if path is None:
            path = ""
        else :
            pass


        for image in list_image:
            image_hold = cv2.imread(str(path)+"/"+image,0)
            self.templade_match.append(image_hold)

        print(self.templade_match)

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


        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
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

        self.showImage()

        gray = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)

        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        reactKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        gradX = cv2.morphologyEx(self.image,cv2.MORPH_CLOSE,sqKernel)
        self.image = gradX
        self.showImage()

        dilate = cv2.dilate(gradX,cv2.getStructuringElement(cv2.MORPH_OPEN,(4,4)),iterations=1)

        self.image = dilate

        rat ,thresh = cv2.threshold(dilate,90,255,cv2.THRESH_BINARY)

        kernel1 = numpy.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], numpy.uint8)
        
        kernel2 = numpy.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], numpy.uint8)
        

        morplot = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
        # self.save('teste1.png')/
        im2, cnts , hierarchy = cv2.findContours(morplot.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(self.image,cnts,-1,(0,255,0),3)

        self.showImage()

        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
        
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

            
            # roi = cv2.dilate(roi,(-2,-2))

            # print('---------------------')
            # print(roi)
            # print('---------------------')
            # dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,rex)

            # print('--------------------- hoh')
            # print(hoi_hog_fd)
            # print('--------------------- hoh')

            print("rect 2 {0}".format(rect[2]))

            if (int(rect[3]) > 60):
                cv2.imshow('image',roi)
                cv2.waitKey(0)
                roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)

                hoi_hog_fd = hog(roi,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
                self.images.append(roi)
                predict = self.model.predict(numpy.array([hoi_hog_fd], 'float64'))
                cv2.putText(self.image, str(int(predict)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

            # cv2.imshow('image',roi)
            # nbr = self.clf.predict(numpy.array([hoi_hog_fd],'float64'))
            # predicts.append(nbr)
        # cv2.imshow("machine",gray)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.showImage()
        exit()
        # model_s = LinearSVC()
        # model_s.fit(self.images,numpy.array([0,0,6,9,4,9,3,8,1,9,0]))
        new_featues = []
        for feature in self.images:
            img = hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualise=False)
            new_featues.append(img)

        # print(new_featues)
        new_featues = numpy.array(new_featues,'float64')

        self.pushData(new_featues,numpy.array([1,0,9,7,7,1,5,3,2,1,0]))

        exit()
        # marc = [0,1,1,1,2,7,2,9,7,0,1]
        # df = pandas.DataFrame(new_featues)
        # df['data'] = new_featues
        # df['target'] = marc
        # print(df)
        # self.df.append(df)
        # self.df.to_csv("fileMachine1.csv")
        # self.df = pandas.concat([self.df,df],sort=True)
        # print(self.df)
        # df.to_csv('fileMachine4.csv',index=False)

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
        model.fit(new_featues,numpy.array([0,7,7,4,3,1,3,7,7,0,0]))

        # joblib.dump(new_featues,'digits_features.pkl',compress=3)
        # print(model.target)

        print('----------- 2')

        for rect in reacts:
            cv2.rectangle(self.image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
            leng = int(rect[3] * 1)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = thresh[pt1:pt1+leng, pt2:pt2+leng]

            # if (rect[2] >= 10 and rect[2] <= 80) and (rect[3] >= 30 and rect[3] <= 90):

            # print("react 3 {0}".format(rect[3]))

            cv2.imshow('image',roi)
            cv2.waitKey(0)
            # print('--------------------- hoh')
            # print(hoi_hog_fd)
            # print('--------------------- hoh')

            # if (rect[3] >= 50):
                # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                # roi = cv2.dilate(roi,(-2,-2))
                # print('---------------------')
                # print(roi)
                # print('---------------------')
                # dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,rex)
                # cv2.imshow('image', roi)
                # cv2.waitKey(0)
                # hoi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

                # if quantaty_fe == len(hoi_hog_fd):
                #     pre = model.predict(numpy.array([hoi_hog_fd],'float64'))
                #     print(pre)

        # model.fit(new_featues,numpy.array([0,0,0]))
        cv2.destroyAllWindows()
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


    def imageFraca(self):
        gray  = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        ret , binary = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

        binary = cv2.bitwise_not(binary)

        H = cv2.Sobel(binary,cv2.CV_8U,0,2)
        V = cv2.Sobel(binary,cv2.CV_8U,2,0)

        rows , cols = self.image.shape[:2]

        _ , countours , _ = cv2.findContours(V,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in countours :
            (x,y,w,h) = cv2.boundingRect(cnt)

            if h > rows/4:
                cv2.drawContours(V,[cnt],-1,255,-1)
                cv2.drawContours(binary,[cnt],-1,255,-1)
            else:
                cv2.drawContours(V,[cnt],-1,0,-1)

        _ , contours , _ = cv2.findContours(H,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (w,y,w,h) = cv2.boundingRect(cnt)

            if w > cols / 3:
                cv2.drawContours(H,[cnt],-1,255,-1)
                cv2.drawContours(binary,[cnt],-1,255,-1)
            else :
                cv2.drawContours(H,[cnt],-1,0,-1)
        
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(3,3))
        H = cv2.morphologyEx(H,cv2.MORPH_DILATE,kernel,iterations=3)
        V = cv2.morphologyEx(V,cv2.MORPH_DILATE,kernel,iterations=3)

        cv2.imshow("himage",H)
        cv2.waitKey(0)

        cv2.imshow("wimage",V)
        cv2.waitKey(0)

        self.showImage()
        # cv2.destroyAllWindows()
        # for cnt in contours:
        #     (x,y,w,h) = 


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
# img = ScannerDespachante(image='images/Scanner_20180820 (5)-1.png')
# img.imageFraca()
# img.forceHSV()
# img.findContours()
# img.findContours()
# img.black()
# img.react()
# img.inRange([0,0,0],[140,140,140],mask=False)
# img.showImage()
# img.green()
# img.black()
# img.blackandwrite()
# img.showImage()
# img.threshold()
# img.dilate()
# img.morphologyEx()
# img.dilate()
# img.showImage()
# data = pandas.read_csv("cvfile.csv",names=['data','target'])
# print(data[:,data.columns])
# df = pandas.DataFrame([1,2,32],columns=['data'])
# print(df)
# df['target'] = marc

# df.to_csv('fileMachine.csv',index=False)