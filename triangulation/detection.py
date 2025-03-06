import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

       
        self.subscription1 = self.create_subscription(
            Image,
            'camera1/image_raw',
            self.image_callback1,
            10
        )
        self.subscription2 = self.create_subscription(
            Image,
            'camera2/image_raw',
            self.image_callback2,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Object Detection Node Started")

       
        prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
        model_path = 'models/MobileNetSSD_deploy.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        
        self.colours = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.min_confidence = 0.2  # 置信度阈值

    def detect_objects(self, frame):
        targets = []
        height, width = frame.shape[:2]

       
        blob_img = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007, (300, 300), 130)
        self.net.setInput(blob_img)
        detected_objects = self.net.forward()

        for i in range(detected_objects.shape[2]):
            confidence = detected_objects[0, 0, i, 2]
            if confidence > self.min_confidence:
                class_index = int(detected_objects[0, 0, i, 1])
                upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                prediction_text = f"{self.classes[class_index]}: {confidence:.2f}%"
                cv2.rectangle(frame, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y),
                              self.colours[class_index], 3)
                cv2.putText(frame, prediction_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colours[class_index], 2)

                
                targets.append([self.classes[class_index], upper_left_x, upper_left_y, lower_right_x, lower_right_y])

        return frame, targets

    def image_callback1(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame, targets = self.detect_objects(frame)
            cv2.imshow("Camera 1 Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing Camera 1 frame: {e}")

    def image_callback2(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame, targets = self.detect_objects(frame)
            cv2.imshow("Camera 2 Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing Camera 2 frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
