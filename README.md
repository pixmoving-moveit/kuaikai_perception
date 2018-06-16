# kuaikai_perception
Detection of traffic signs for fuel stop and bus stop, vehicles, and pedestrians

## Sign and Checkerboard detection
Dependencies:
1. scikit-learn v0.14+, numpy v1.14+
2. OpenCV3 with opencv_contrib installed, and Python bindings
3. ROS Kinetic or above, rospy

Usage:
> python detectsign_ros.py

Subscribes to two camera image topics: /camera0/image_raw and /camera1/image_raw. Takes the bottom right quarter of the image for simplifying computations and also improving detection accuracy. Uses redundancy to ensure both cameras are identifying the same sign. Publishes to the /light_color topic. This may need to be changed to the /light_color_managed topic, and may also need some "hacking" to ensure the car obeys the command.
