import cv2
import numpy as np
from copy import copy
from skimage.measure import compare_ssim
import time

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError



class image_reader:
    def __init__(self):
        #Create publisher
        self.pub = rospy.Publisher("/light_color", Int32, queue_size = 0)

        self.bridge = CvBridge()
        self.image_top = rospy.Subscriber('/camera0/image_raw', Image, self.callback)
    print('Subscribed to top camera')
        self.image_front = rospy.Subscriber('/camera1/image_raw', Image, self.callback)
    print('Subscribed to front camera')
   
    self.redsign = []
    self.redcheck = []
    self.is_check = False
    self.is_sign = False

    def callback(self, data):
       
        try:
            image = self.bridge.imgmsg_to_cv2(data)

            #Get checkerboard images from both cameras 
            start = self.checkerboard(image)
            self.redcheck.append(start)
           
            #Check for redundancy from both cameras in checkerboard detection
            if(len(self.redcheck)==2):
                if(self.redcheck[0]==self.redcheck[1] and self.redcheck[0]==1):
                    self.is_check = True
                    print "found checkerboard"
                else:
                    self.is_check = False
                self.redcheck=[]

            #Get sign images from both cameras
            sign = self.signdetection(image)
            print sign
            self.redsign.append(sign)
           
            #Check for redundancy from both cameras in sign detection           
            if(len(self.redsign) == 2):
                if(self.redsign[0] == self.redsign[1] and self.redsign[0]==1):
                    self.is_sign = True    #fueltank found, so stop -> RED=0
                    print "found sign"
                else:
                    self.is_sign=False
                self.redsign = []
               
            #Check for either sign or checkboard to appear
            if(self.is_check==True or self.is_sign ==True):
                self.pub.publish(0)
            else:
                self.pub.publish(1)
            self.is_sign = False
            self.is_check = False
               
            cv2.waitKey(3)
           
        except:
            return('error')

    def signdetection(self, frontcam):
        frontcam = cv2.cvtColor(frontcam, cv2.COLOR_RGB2BGR)[1024:,1024:,:]
       
        fronthsv = cv2.cvtColor(frontcam, cv2.COLOR_BGR2HSV)
       
        lower_blue = np.array([100,150,100])
        upper_blue = np.array([130,255,255])

        frontmask = cv2.inRange(fronthsv,lower_blue,upper_blue)
       
        kernel=np.ones((5,5),np.uint8)

        frontmask = cv2.morphologyEx(frontmask, cv2.MORPH_CLOSE, kernel)
       
        frontout = cv2.bitwise_and(frontcam,frontcam,mask=frontmask)
        ########### EXTRACT THE BLOBS ############
       
        _, frontcont, hierarchy = cv2.findContours(frontmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
        frontcont = sorted(frontcont, key=cv2.contourArea, reverse=True)
        if len(frontcont)>0:
            if cv2.contourArea(frontcont[0])>200:
                #print "insidefn"

                x,y,w,h=cv2.boundingRect(frontcont[0])
                if float(w)/float(h)>1.2 or float(w)/float(h)<0.8:
                    return -1
                frontcrop=frontcam[y+5:y+h-5, x+5:x+w-5, :].copy()
                cv2.rectangle(frontcam,(x+5,y+5),(x+w-5,y+h-5),(0,255,0),3)

                bustemp=cv2.imread("busstop.jpg",0)
                bustemp=cv2.resize(bustemp,(80,80),0,0,cv2.INTER_NEAREST)
                fueltemp=cv2.imread("fueltank.jpg",0)
                fueltemp=cv2.resize(fueltemp,(80,80),0,0,cv2.INTER_NEAREST)
                pedtemp=cv2.imread("pedestrian.jpg",0)
                pedtemp=cv2.resize(pedtemp,(80,80),0,0,cv2.INTER_NEAREST)
               
                frontcrop=cv2.cvtColor(frontcrop,cv2.COLOR_BGR2GRAY)
                frontcrop=cv2.resize(frontcrop,(80,80),0,0,cv2.INTER_NEAREST)

                frontscorebus, frontdiffbus = compare_ssim(bustemp,frontcrop, full=True)
                frontscorefuel, frontfuel = compare_ssim(fueltemp,frontcrop, full=True)
                frontscoreped, frontped = compare_ssim(pedtemp,frontcrop, full=True)
                #print frontscorebus,frontscorefuel
                if frontscorebus>frontscoreped and frontscorebus>frontscorefuel:
                    #print "The camera says bus"
                    return 0
                else:
                    #print "The camera says fuel"
                    return 1
        #print "No sign found"
        return -1

    def checkerboard(self, im):
        kernel = np.ones((3,3),np.uint8)
        im = im[1024:,1024:,:]
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im,(5,5),0)
        im = cv2.medianBlur(im,5)
        th2=cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
        #show(th2)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        params.filterByArea = True
        params.minArea = 100

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints=detector.detect(th2)
        if 25<len(keypoints)<40:
            return 1            #checkerboard found, so stop -> publish 0
        return -1
       

def main():
    imager = image_reader()
    rospy.init_node('detection')
    try:
        rospy.spin()
    except:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
