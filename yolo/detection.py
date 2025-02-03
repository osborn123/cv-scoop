import cv2
import torch
from ultralytics import YOLO

# 载入 YOLOv5 预训练模型
model = YOLO("yolov8n.pt")  # 你可以换成更大的模型 "yolov8m.pt"

# 读取图像
image_path = "image.jpg"  # 替换成你的图片路径
image = cv2.imread(image_path)

# 进行物品识别
results = model(image)

# 显示检测结果
for result in results:
    boxes = result.boxes  # 获取检测框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取框的坐标
        label = model.names[int(box.cls[0])]  # 物体类别
        conf = float(box.conf[0])  # 置信度

        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图片
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
