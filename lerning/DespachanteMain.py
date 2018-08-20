import imutils
import numpy as np
import cv2

from skimage import filters , io as shio
import matplotlib.pyplot as plt

m = ScannerDespachante(image='n.png')
m.blacks()
m.showImage()
# kernel = np.ones((6,6),np.float32) / 25
# dst = cv2.filter2D(m,-1,kernel)

# dst = cv2.GaussianBlur(m,(3,3),0)

dst = cv2.bilateralFilter(m,9,75,75)

cv2.imshow('ma',np.hstack([dst,m]))
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()




fgbg = cv2.createBackgroundSubtractorMOG2()

gray = cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,11,17,17)
edged = cv2.Canny(gray,30,300)

cv2.imshow('ma',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()

rgb_screem = cv2.cvtColor(m,cv2.COLOR_BGR2RGB)

lower = np.array([72,160,160])
upper = np.array([112,247,255])

mask = cv2.inRange(rgb_screem,lower,upper)
ouput = cv2.bitwise_and(rgb_screem,rgb_screem,mask=mask)

cv2.imshow('ma',np.hstack([ouput]))
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

b[(( b > g) & (b -r > 30 ))] = 255

img = cv2.merge((b,g,r))

cv2.imshow('ma',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()



scanner = ScannerDespachante('teste-2-2.png')

scanner.green()

scanner.black()
scanner.black()
scanner.black()

scanner.blackandwrite()

scanner.brightness(1.0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)

scanner.brightness(1.0)

scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)
scanner.constrast(30,0)

scanner.blacks()

scanner.save(name='scannerjon.png')

scanner.showImage()








