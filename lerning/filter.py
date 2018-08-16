import numpy
import cv2
from DOC import imageTesseract
from skimage.filters import threshold_local
img = cv2.imread('renavam.jpeg')
#
try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract
# raw = img.copy()
#
# img[numpy.where((img >= [200,240,200]).all(axis=2))] = [201,234,184]
# img[numpy.where((img == [104,122,93]).all(axis=2))] = [0,0,0]
#
# ratio = img.shape[0] / 500.0
# orig = img.copy()
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 75, 200)
#
# warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# t = threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > t).astype("uint8") * 255
#
# cv2.imwrite('tesse.jpeg',warped)

img = Image.open('nu.jpg')
text = pytesseract.image_to_string(img, lang='eng', boxes=False,config='digits -psm 7')
print(text)
# lower = numpy.array([255,255,255],dtype="uint16")
# upper = numpy.array([110,110,110],dtype="uint16")
# black_mask = cv2.inRange(img,lower,upper)
# cv2.imshow('mask',black_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()