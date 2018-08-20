import matplotlib
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

exit()

# img = cv2.imread('scds.png')
#
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = cv2.imwrite('renagray.png',img)


img = cv2.imread('renacod.png')
img = numpy.int32(img)

constrant = 64
brightness = 0

img = img * (constrant / 127 + 1) - constrant + brightness
img = numpy.clip(img, 0, 255)
img = numpy.clip(img, 0, 255)
img = numpy.uint8(img)

def rgb(img,color):

    (B,G,R) = cv2.split(img)
    M = numpy.maximum(numpy.maximum(R,G),B)

    R[ G < R] = color
    G[ G > B ] = color
    R[ R < 190 ] = color

    return  cv2.merge([B,G,R])


def render(image,name):
    for i in range(0,1):
        (B,G,R) = cv2.split(image)
        # print(( numpy.all(image < [190,190,190]) and numpy.all(G > R)))
        print('___________')
        # print(image[:,:,1] > image[:,:,2])]]
        b , g , r = image[...,0] , image[...,1] , image[...,2]

        # image[( g < 120 ) & (r < 190) & (g < 190)] = [0,0,0]

        # cv2.imwrite('test1.jpg',image)

        image[((b > g) & (b > r))] = [255, 255, 255]

        cv2.imwrite(filename=name,img=image)

        image = cv2.imread(name)

        # imgs = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image[((g > r) & (g > b)) & ((r > 100))] = [255, 255, 255]

        cv2.imwrite(filename='verde.jpg',img=image)

        # image[((g != 0) & (r != 0)) & ((b != 0))] = [255, 255, 255];
        #
        # cv2.imwrite('test1.jpg',image)


        # print(image)
        #
        # print('___________________ssss')
        #
        # for a in image:
        #     b , g , r = a[...,0] , a[...,1] , a[...,2]
        #
        #     print(image[a])
        #
        #     exit()
        # if numpy.where(g > r) and numpy.where(g > b) and numpy.where(g < 190) and numpy.where(r < 190) and numpy.where(b < 190):
        #     image[a] = [0,0,0]
        # else :
        #     image[a] = [25,255,255]

        # cv2.imwrite('testej.png',image)
        # print(colors)

        # print(numpy.where(image[:,:,2] < image[:,:,1] & image[:,:,1] > image[:,:,2]))
        # print( numpy.where(image < [190,190,190] and image[:,:,1] > image[:,:,2] and image[:,:,1] > image[:,:,0]).any())
        # cv2.imwrite('renacod{0}.png'.format(i),img=image)

# img = render(cv2.imread('r.png'),'test1.png')
# rena = render(cv2.imread('test1.jpg'),'test2.png')
b  = render(cv2.imread('n.png'),'test3.png')
# print(numpy.where(numpy.ndarray((1,2)) > numpy.ndarray((3,2))))

# h = numpy.array([1,2,3,2])

# print(h[ h == 2 ])