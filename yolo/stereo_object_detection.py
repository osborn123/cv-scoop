import cv2
import numpy as np
from ultralytics import YOLO

# 加载相机参数
params = np.load("stereo_params.npz")
mtxL, distL, mtxR, distR, R, T = params["mtxL"], params["distL"], params["mtxR"], params["distR"], params["R"], params["T"]

# 计算立体校正
stereo_rectify = cv2.stereoRectify(mtxL, distL, mtxR, distR, (640, 480), R, T)
R1, R2, P1, P2, Q = stereo_rectify[:5]

# 立体匹配（计算视差图）
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=15,
    P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2,
    disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)

# 设定基线距离（单位 cm）
BASELINE = 6.0
FOCAL_LENGTH = P1[0, 0]

# 加载 YOLO 物品识别模型
model = YOLO("yolov8n.pt")

# 打开双目摄像头
capL = cv2.VideoCapture(0)  # 左摄像头
capR = cv2.VideoCapture(1)  # 右摄像头

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # 计算视差图
    disparity_map = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 计算深度
    depth_map = (FOCAL_LENGTH * BASELINE) / (disparity_map + 0.0001)

    # 物品识别
    results = model(frameL)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # 计算物体中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 读取该点的深度
            if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
                distance = depth_map[center_y, center_x]
            else:
                distance = -1  # 无效测距

            # 绘制检测框和距离信息
            cv2.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frameL, f"{label} {conf:.2f} Dist:{distance:.2f}cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Stereo Object Detection & Distance Measurement", frameL)
    cv2.imshow("Disparity", disparity_map / np.max(disparity_map))
    cv2.imshow("Depth", depth_map / np.max(depth_map))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
