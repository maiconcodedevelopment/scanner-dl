import cv2
import numpy
import matplotlib
from imutils.perspective import four_point_transform

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

    def threshold(self):
        ret , self.image = cv2.threshold(self.image,127,255,cv2.THRESH_BINARY)

    def rectangle(self):
        cv2.rectangle(self.image,(100,100),(3,40),(255,255,0),2)

    def blacks(self):
        kernel = numpy.ones((3, 3), numpy.uint8)
        dilation = cv2.dilate(self.image, kernel, iterations=1)
        self.image = dilation


    def black(self):
        self.image[( self.g < 120 ) & (self.r < 190) & (self.b < 190)] = [0,0,0]

    def blur(self):
        self.image = cv2.bilateralFilter(self.image,9,75,75)

    def mediablur(self):
        self.image = cv2.medianBlur(self.image,5)

    def erode(self):
        self.image = cv2.erode(self.image,numpy.zeros((5,5),numpy.uint8),iterations=1)