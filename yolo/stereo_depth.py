import cv2
import numpy as np

# 加载相机参数
params = np.load("stereo_params.npz")
mtxL, distL, mtxR, distR, R, T = params["mtxL"], params["distL"], params["mtxR"], params["distR"], params["R"], params["T"]

# 计算立体校正矩阵
stereo_rectify = cv2.stereoRectify(mtxL, distL, mtxR, distR, (640, 480), R, T)
R1, R2, P1, P2, Q = stereo_rectify[:5]  # Q矩阵用于深度计算

# 立体匹配算法
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=15,
    P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2,
    disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)

# 打开摄像头
capL = cv2.VideoCapture(0)  # 左摄像头
capR = cv2.VideoCapture(1)  # 右摄像头

# 设定基线距离（单位 cm）
BASELINE = 6.0
FOCAL_LENGTH = P1[0, 0]  # 获取焦距

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # 计算视差图
    disparity_map = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 计算深度图
    depth_map = (FOCAL_LENGTH * BASELINE) / (disparity_map + 0.0001)  # 避免除零

    # 归一化显示
    cv2.imshow("Disparity", disparity_map / np.max(disparity_map))
    cv2.imshow("Depth", depth_map / np.max(depth_map))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
