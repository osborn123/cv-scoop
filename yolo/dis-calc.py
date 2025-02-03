import cv2
import numpy as np

# 设定物体的实际宽度（单位：cm）
KNOWN_WIDTH = 10.0  # 例如你知道目标物体宽度是10cm
KNOWN_DISTANCE = 50.0  # 设定一个已知的测距距离（单位：cm）

# 计算焦距（需要先进行标定）
FOCAL_LENGTH = 500  # 这里假设一个焦距，你可以用标定方法计算

# 读取图像
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150)

# 进行轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)  # 计算包围盒
    if w > 50:  # 过滤掉小的噪声
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w  # 计算距离
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图片
cv2.imshow("Distance Measurement", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
