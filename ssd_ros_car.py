'''
-Algorithm to follow front vehicle-

Working principle:
    -Check if host vehicle is in region of interest and in Advance Challenge.
    -Check if vehicle-to-follow is in front of host vehicle.
    -Estimate if vehicle-to-follow is close or far. According to this publish stop or go.

TODO: Check working principle. (This code was never tested in real car)
TODO: Algorithm will detect car, but distance approximation is still not completed.
TODO: Region of Interest (ROI) of where the algorithm should start to publish needs to be finished.
'''
import cv2
import numpy as np
from copy import copy
import time
import os

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf
from keras.optimizers import Adam
from ssd_keras.models.keras_ssd300 import ssd_300
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss

# Root directory of the project
global ROOT_DIR
ROOT_DIR = os.getcwd()

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class car_detector:
    def __init__(self):
        #Create regions of interest
        self.x_min_follow = 148.0
        self.x_max_follow = 168.0
        self.y_min_follow = -6.81
        self.y_max_follow = 118.0

        self.inside_follow_roi = False
        self.count_crossed_pedestrian_roi = 0
        self.advance_challenge = False

        #Create publisher
        self.pub = rospy.Publisher("/light_color", Int32, queue_size = 0)

        #Create car postition subscriber
        self.car_position = rospy.Subscriber('/current_pose', PoseStamped ,self.callback_pose)

        #Create image subscriber
        self.bridge = CvBridge()
        self.image_top = rospy.Subscriber('/camera0/image_raw', Image, self.callback_img)
        print('Subscribed to top camera')

        img_height = 300
        img_width = 480
        img_channels = 3
        #The per-channel mean of the images in the dataset
        subtract_mean = [123, 117, 104]
        #The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
        swap_channels = [2, 1, 0]
        #Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
        n_classes = 1
        scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
        #The anchor box aspect ratios used in the original SSD300; the order matters
        two_boxes_for_ar1 = True
        steps = [8, 16, 32, 64, 100, 300]
        #The space between two adjacent anchor box center points for each predictor layer.
        offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        #The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
        clip_boxes = False
        #Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
        variances = [0.1, 0.1, 0.2, 0.2]
        #The variances by which the encoded target coordinates are scaled as in the original implementation
        normalize_coords = True

        #Create model and load weights
        self.model = ssd_300(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_per_layer=aspect_ratios,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=subtract_mean,
                        divide_by_stddev=None,
                        swap_channels=swap_channels,
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400,
                        return_predictor_sizes=False)
        print("Model built.")

        try:
            #Load the sub-sampled weights into the model.
            weights_path = ROOT_DIR + '/weights/VGG_coco_SSD_300x300_iter_400000_car.h5'
            self.model.load_weights(weights_path, by_name=True)
            print("Weights file loaded:", weights_path)
        except:
            print("Weights not found")

        #Instantiate an Adam optimizer and the SSD loss function and compile the model.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        self.graph = tf.get_default_graph()

    def callback_pose(self, data):
        self.x = data.pose.position.x
        self.y = data.pose.position.y

    def callback_img(self, data):
        image = self.bridge.imgmsg_to_cv2(data)
        prediction = self.detect_car(image)
        '''
        Note that the car will cross the region of interest multiple times so
        we need to count how many times you cross the region of interest or use the
        pedestrian roi to know if you are already in advanced challenge.
        If required, pedestrian ROI needs to be defined (TODO).
        '''
        #Count number of times you cross roi for pedestrian crossing, if n==3 means you are in
        #advance challenge
        if ((self.x_min_ped < self.x) and (self.x < self.x_max_ped)):
            if ((self.y_min_follow < self.y) and (self.y < self.y_max_ped)):
                self.count_crossed_pedestrian_roi +=1

        if self.count_crossed_pedestrian_roi == 3):
            self.count_crossed_pedestrian_roi = 0
            self.advance_challenge = True

        #Only publish if you are inside region and during advance challenge
        if ((self.x_min_follow < self.x) and (self.x < self.x_max_follow)) and self.advance_challenge:
            if ((self.y_min_follow < self.y) and (self.y < self.y_max_follow)):
                self.inside_follow_roi = True

        if (self.inside_follow_roi):
            print('Danger, you are in roi to follow car')
            if (close_to_followed):
                self.pub.publish('Stop or reduce velocity')
            if (far_to_followed):
                self.pub.publish('You can continue at your default speed')
            self.advance_challenge = False
            self.inside_follow_roi = False


        if len(prediction[0]) >= 1:
            print("Car in front")
        else:
            print("No car in front")
        cv2.waitKey(5)


    def detect_car(self, img):
        batch_images = []
        image = cv2.resize(img, (480,300), interpolation = cv2.INTER_AREA)
        batch_images.append(image)
        batch_images = np.array(batch_images)
        with self.graph.as_default():
            y_pred = self.model.predict(batch_images)
        # Decode the raw prediction.
        i = 0
        confidence_threshold = 0.9
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        return(y_pred_thresh)

def main():
    pedestrian = car_detector()
    rospy.init_node('car_detection')
    try:
        rospy.spin()
    except:
        print('Shutting Down')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
