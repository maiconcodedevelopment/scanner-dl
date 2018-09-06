import urllib.request
import cv2
import numpy
import os

def store_raw_images():
    neg_images_link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152"
    neg_images_urls = urllib.request.urlopen(neg_images_link).read().decode()

    if not os.path.exists("neg"):
        os.makedirs("neg")

    pic_num = 1
    
    for i in neg_images_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i,"neg/"+str(pic_num)+'.jpg')
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num +=1
            
        except Exception as identifier:
            print(str(identifier))


def find_uglies():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir("uglies"):
                try:
                    current_image_path = str(file_type)+"/"+str(img)
                    ugly = cv2.imread("uglies/"+str(ugly))
                    question = cv2.imread(current_image_path)

                    if ugly.shape == question.shape and not(numpy.bitwise_xor(ugly,question).any()):
                        print("predict image path dectect !!!")
                        os.remove(current_image_path)

                except Exception as identifier:
                    print(str(identifier))

def create_pos_n_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):
            if file_type == "neg":
                line = file_type+"/"+img+"\n"
                with open("bg.txt","a") as f:
                    f.write(line)

            elif file_type == "pos":
                line = file_type+"/"+img+" 1 0 0 50 50\n"
                with open("info.dat","a") as f:
                    f.white(line)

if __name__ == "__main__":
    #tutorial examples : https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/
    store_raw_images()
    # find_uglies()
    # create_pos_n_neg()

    # watch_cascade = cv2.CascadeClassifier("cascade.xml")

    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret , img = cap.read()
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     watches = watch_cascade.detectMultiScale(gray,120,120)

    #     for (x,y,w,h) in watches:
    #         print("sim")
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        
    #     cv2.imshow("img",img)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
    # implemtação de seguro progressimo