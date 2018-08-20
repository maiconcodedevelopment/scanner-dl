import cv2
import numpy

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
        
