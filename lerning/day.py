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

import cv2
import numpy as npThank

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

# Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

        box_extraction("41.jpg", "./Cropped/")