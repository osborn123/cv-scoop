from ultralytics import YOLO
import cv2

# 载入模型
model = YOLO("yolov8n.pt")

# 设定已知物体的实际宽度（单位：cm）
KNOWN_WIDTH = 10.0
FOCAL_LENGTH = 500  # 需要标定

# 读取图像
image = cv2.imread("image.jpg")
results = model(image)

# 处理检测结果
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        
        # 计算物体宽度（像素）
        pixel_width = x2 - x1
        if pixel_width > 0:
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width  # 计算距离

            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f} Dist:{distance:.2f}cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图片
cv2.imshow("Object Detection & Distance Measurement", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
