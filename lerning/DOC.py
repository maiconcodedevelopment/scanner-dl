import numpy
import cv2
import sys
import pytesseract
import imutils
from imutils import contours
import argparse
from skimage.filters import threshold_local
from matplotlib import pyplot as plt

def emplyFunction():
    pass

def main():

    img = numpy.zeros((100,100,3),numpy.uint8)
    cv2.namedWindow('rena')

    def clickWindows(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),40,(0,255,0),-1)
        # if event == cv2.EVENT_LBUTTONDOWN:


    print(numpy.uint8)
    # cv2.createTrackbar('B','rena',0,255,emplyFunction)
    # cv2.createTrackbar('G','rena',0,255,emplyFunction)
    # cv2.createTrackbar('R','rena',0,255,emplyFunction)

    cv2.setMouseCallback('rena',clickWindows)

    while (True):
        cv2.imshow('rena',img)

        if cv2.waitKey(1) == 27:
            break

        # blue =  cv2.getTrackbarPos('B','rena')
        # green = cv2.getTrackbarPos('G','rena')
        # red =   cv2.getTrackbarPos('R','rena')
        #
        # img[:] = [blue,green,red]
        # print(blue,green,red)

    cv2.destroyAllWindows()


def eventsCv2():

    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)

def imageTesseract():

    renavam = cv2.imread('scds.png')

    r = cv2.imread('scds.png',0)
    r = cv2.medianBlur(r,5)

    ret,th1 = cv2.threshold(r,126,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    titles = [
        'Original','Gloabal Image','Adaptive','Apative Gausi'
    ]

    images = [r,th1,th2,th3]

    for i in range(0,4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

    ratio = renavam.shape[0] / 500.0
    orig = renavam.copy()

    gray = cv2.cvtColor(renavam,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edged = cv2.Canny(gray,75,200)

    warped = cv2.cvtColor(renavam,cv2.COLOR_BGR2GRAY)
    t = threshold_local(warped,11,offset=10,method="gaussian")
    warped = (warped > t).astype("uint8") * 255



    cv2.imshow("image",renavam)
    cv2.imshow("edged",edged)
    cv2.imshow("scanned",warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cameraRead():

    camera = cv2.VideoCapture(0)
    while(True):

        (grabbed,frame) = camera.read()

        skin = cv2.bitwise_and(frame,frame)

        cv2.imshow('images',numpy.hstack([frame,skin]))

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


def cameraColor():

    frame = cv2.imread('we.jpg')

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = numpy.array([100,50,50])
    upper_red = numpy.array([18,255,255])

    mask = cv2.inRange(hsv,lower_red,upper_red)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
        # dark_red = numpy.uint8([[[12,22,121]]])
        # dark_red = cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)

if __name__ == "__main__":

    imageTesseract()
    # im = cv2.imread('scds.png')
    #
    # hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    #
    # high = numpy.array([100,50,50])
    # low = numpy.array([193,229,176])
    #
    # image_mask = cv2.inRange(hsv,low,high)
    # output = cv2.bitwise_and(im,im,mask=image_mask)
    #
    # cv2.imshow('image', image_mask)
    # cv2.imshow('origin', im)
    # cv2.imshow('color trick', output)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


