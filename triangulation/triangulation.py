import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from frame_angle import Frame_Angle
import detection as dt

class TriangulationNode(Node):
    def __init__(self):
        super().__init__('triangulation_node')

        self.subscription1 = self.create_subscription(
            Image, 'camera1/image_raw', self.image_callback1, 10)
        self.subscription2 = self.create_subscription(
            Image, 'camera2/image_raw', self.image_callback2, 10)

        self.bridge = CvBridge()
        self.angler = Frame_Angle(1280, 1024, 63, 46)
        self.camera_separation = 10.8
        self.frame1 = None
        self.frame2 = None

    def image_callback1(self, msg):
        self.frame1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.process_frames()

    def image_callback2(self, msg):
        self.frame2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.process_frames()

    def process_frames(self):
        if self.frame1 is None or self.frame2 is None:
            return

        targets1 = dt.detection(self.frame1)
        targets2 = dt.detection(self.frame2)

        if len(targets1) == 0 or len(targets2) == 0:
            return

        x1m = (targets1[0][3] - targets1[0][1]) // 2 + targets1[0][1]
        y1m = (targets1[0][4] - targets1[0][2]) // 2 + targets1[0][2]
        x2m = (targets2[0][3] - targets2[0][1]) // 2 + targets2[0][1]
        y2m = (targets2[0][4] - targets2[0][2]) // 2 + targets2[0][2]

        xlangle, ylangle = self.angler.angles_from_center(x1m, y1m, top_left=True, degrees=True)
        xrangle, yrangle = self.angler.angles_from_center(x2m, y2m, top_left=True, degrees=True)

        X, Y, Z, D = self.angler.location(self.camera_separation, (xlangle, ylangle), (xrangle, yrangle), center=True, degrees=True)
        self.get_logger().info(f"Target Position: X={X}, Y={Y}, Z={Z}, D={D}")

rclpy.init()
node = TriangulationNode()
rclpy.spin(node)
