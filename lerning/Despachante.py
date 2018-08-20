import cv2
import numpy
import matplotlib
from imutils.perspective import four_point_transform
import imutils
import pytesseract
from PIL import Image

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
        self.name = name

        self.b = self.image[...,0]
        self.g = self.image[...,1]
        self.r = self.image[...,2]
        
        self.count = 0
        self.hstack = None
        self.mask = ""

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
        self.count = self.count + 1
        cv2.namedWindow('imagem'.format(self.count),cv2.WINDOW_AUTOSIZE)
        cv2.imshow('imagem'.format(self.count),numpy.hstack([self.image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def green(self):
        b, g, r = self.image[..., 0], self.image[..., 1], self.image[..., 2]
        self.image[((b > g) & (b > r) & ((b - r) > 30))] = [255, 255, 255]

    # def threshold(self):
    #     ret , self.image = cv2.threshold(self.image,127,255,cv2.THRESH_BINARY)
    
    def rectangle(self):
        cv2.rectangle(self.image,(100,100),(3,40),(255,255,0),2)

    def dilate(self):
        kernel = numpy.ones((25, 25), numpy.uint8)
        dilation = cv2.dilate(self.mask, kernel, iterations=1)
        opening = cv2.morphologyEx(self.mask,cv2.MORPH_CLOSE,kernel)
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
        self.mask = cv2.inRange(self.image,numpy.array(lower,dtype='uint16'),numpy.array(upper,dtype='uint16'))
        self.dilate()
        if mask :
            self.image = cv2.bitwise_and(self.image,self.image,mask=self.mask)

        # if mask :
    def threshold(self):
        print( cv2.THRESH_BINARY )
        ret , self.image = cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY_INV)

        
    def findContours(self):

        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,11,17,17)
        edged = cv2.Canny(gray,50,200,255)
        
        cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]  if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
        displayCnt = None

        # for c in cnts:-
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


    img = ScannerDespachante(image='scanner.png')
    img.inRange([0,0,0],[100,100,100],mask=True)
    img.threshold()
    img.showImage()