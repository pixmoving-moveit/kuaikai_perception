
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

class pedestrian_detector:
    def __init__(self):
        #Create publisher
        self.pub = rospy.Publisher("/light_color", Int32, queue_size = 0)

        #Create image subscriber
        self.bridge = CvBridge()
        self.image_front = rospy.Subscriber('/camera0/image_raw', Image, self.callback_img)
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

        #Load the sub-sampled weights into the model.
        try:
            weights_path = ROOT_DIR + '/weights/VGG_coco_SSD_300x300_iter_400000_pedestrian.h5'
            self.model.load_weights(weights_path, by_name=True)
            print("Weights file loaded:", weights_path)
        except:
            print("Weights not found")

        #Instantiate an Adam optimizer and the SSD loss function and compile the model.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        self.graph = tf.get_default_graph()

    def callback_img(self, data):
        image = self.bridge.imgmsg_to_cv2(data)
        prediction = self.detect_pedestrian(image)
        if len(prediction[0]) >= 1:
            self.pub.publish(0)
            print("Stop, pedestrian in front")
        else:
            self.pub.publish(1)
            print("Go, no pedestrian")

    def detect_pedestrian(self, img):
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
    pedestrian = pedestrian_detector()
    rospy.init_node('pedestrian_detection')
    try:
        rospy.spin()
    except:
        print('Shutting Down')
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
