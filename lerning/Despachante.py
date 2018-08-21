import cv2
import numpy
import matplotlib
from imutils.perspective import four_point_transform
import imutils
from imutils import contours
import pytesseract
from PIL import Image
import skimage


import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from sklearn import datasets
from scipy import ndimage as ndi
from skimage import feature

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

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** invGamma) * 255
                      for i in numpy.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


class ScannerDespachante(object):

    def __init__(self,image="",name="imageDespachange.jpg"):
        self.image = cv2.imread(image)
        self.image1 = self.image

        a = self.image.shape[0]
        l = self.image.shape[1]

        print(a)
        print(l)

        altura = 3000
        largura = int(((altura / ( a / 100)) / 100) * l)

        # print('_____')
        # print(altura)
        # print(largura)
        
        # print("{:10.2f}".format(altura))
        self.showImage()

        image =  cv2.resize(self.image,(largura,altura)) 
        self.image = image
        self.image1 = image
        # self.name = name
        a = self.image.shape[0]
        l = self.image.shape[1]

        # print(a)
        # print(l)
        # m = int((a / 2))
        # self.image = self.image[:m,:]

        rex = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))

        rangerex = cv2.inRange(self.image,numpy.array([50,50,50],dtype='uint16'),numpy.array([200,200,200],dtype='uint16'))
        # threshold = cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),120,255,cv2.THRESH_BINARY_INV)[1]
        gradX = cv2.morphologyEx(rangerex,cv2.MORPH_CLOSE,rex)

        cnts =  cv2.findContours(gradX.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        longs = []

        for (i,c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)
            print("{0} {1}".format(w,h))
            if (w > 1000) & (h > 1000):
                longs.append((x,y,w,h))
                self.image = self.image1[y:y + h,x:x + w]
        # print(rex)
        print(longs)
        # self.image = self.image
        self.showImage()

        exit()
        self.image = self.image[180:245,220:500]
        a = self.image.shape[0]
        l = self.image.shape[1]

        # print(a)
        # print(l)

        altura = 200
        largura = int(((altura / ( a / 100)) / 100) * l)

        self.image = cv2.resize(self.image,(largura,altura))
        self.showImage()
        exit()

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
        self.image[( ( self.g > self.b ) &( self.g > self.r ) & (self.g > 140) )] = [255,255,255]

    # def threshold(self):
    #     ret , self.image = cv2.threshold(self.image,127,255,cv2.THRESH_BINARY)
    
    def rectangle(self):
        cv2.rectangle(self.image,(100,100),(3,40),(255,255,0),2)

    def dilate(self):
        kernel = numpy.ones((3,3), numpy.uint8)
        dilation = None

        if not self.mask is None:
            dilation = cv2.dilate(self.mask, kernel, iterations=1)
            opening = cv2.morphologyEx(self.mask,cv2.MORPH_GRADIENT,kernel)
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

        image = cv2.imread('renavam-1.png')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        reactKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))

        gradX = cv2.morphologyEx(self.image,cv2.MORPH_CLOSE,sqKernel)

        self.image = gradX
        self.showImage()
        exit()

        thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)

        # self.image = thresh
        # self.showImage()
        # self.save('teste1.png')/
        # exit()

        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        locs = []

        for (i ,c )in enumerate(cnts):

            (x,y,w,h) = cv2.boundingRect(c)
            ar = w / float(h)

            print("{0} {1}".format(w,h))
            if ( w > 10 and w < 80) and (h > 10 and h < 45):
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

        

        self.image = thresh

if __name__ == "__main__" :

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

    img = ScannerDespachante(image='images/teste1.png')
    # img.findContours()
    # img.findContours()
    # img.black()
    # img.react()
    # img.inRange([0,0,0],[100,100,100],mask=True)

    # img.green()
    # img.black()
    # img.blackandwrite()
    # img.threshold()
    # img.dilate()
    # img.morphologyEx()

    # img.dilate()
    # img.showImage()