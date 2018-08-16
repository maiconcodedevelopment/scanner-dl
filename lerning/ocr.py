import io
import cv2
import numpy
import matplotlib.pyplot as plt
import  pytesseract
from PIL import Image

# image = cv2.imread('scannner.png',cv2.IMREAD_GRAYSCALE)
# print("{0}".format(image))

# image = Image.open("scanner.png")
# phrage = pytesseract.image_to_string(image=image,lang='por')
# print(phrage)

img = cv2.imread('scanner.png')
print(img.shape)
print(img.item(0,3,2));print(img.item(0,3,1));print(img.item(0,3,0))
img.itemset((0,0,2),255)
img.itemset((0,0,0),0)
img.itemset((0,0,0),0)

# cv2.imwrite('scanner2.png',img)

letra = img[180:260,210:500]
cv2.imwrite("scds.png",letra)

# cv2.namedWindow("DOC Renavam",cv2.WINDOW_AUTOSIZE)
# cv2.imshow('renavam',letra)
# cv2.waitKey(0)
# cv2.destroyAllWindows('renavam')


# 0000000 0x00 0
# 1111111 0xFF 255

img1 = numpy.zeros((100,100,3),numpy.uint8)
windowsName = 'OpenCV BGR Color Patteres'

cv2.namedWindow(windowsName)

while(True):

    cv2.imshow(windowsName,img1)

    if cv2.waitKey(1) == 27:
        break

    cv2.destroyAllWindows()


exit()

n = numpy.ndarray([2,3,3,2]).sum()
print(n)
print(tuple((3,23,34)))

cv2.line(img1,(0,99),(99,0),(255,0,0),2)

                     #largura #altura
cv2.rectangle(img1,(100,100),(3,40),(0,255,0),2)
cv2.circle(img1,(60,60),10,(0,0,255),-1)
cv2.ellipse(img1,(50,60),(50,20),0,0,160,(127,127,127),-1)

points = numpy.array([[80,2],[125,40],[179,19],[230,5],[30,50]],numpy.int32)
points = points.reshape((-1,1,2))

text = 'Test Text'
cv2.putText(img1, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX,2,color=(255,255,0))

cv2.polylines(img1,[points],True,(0,255,255))
cv2.imshow('Lena',img1)
cv2.waitKey(0)
cv2.destroyAllWindows('Lena')

renavam = cv2.imread('scds.png',1)
print(renavam.dtype)
print(renavam.shape)
print(renavam.ndim)
print(renavam.size)

cv2.imshow('rena',renavam)
cv2.imwrite('renavam.jpeg',renavam)
cv2.waitKey(0)
cv2.destroyAllWindows('rena')

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.style.context(style="red",after_reset=True)
# plt.imshow(img,cmap='gray',interpolation='bicubic')
# plt.show()

