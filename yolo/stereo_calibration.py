import cv2
import numpy as np
import glob

CHESSBOARD_SIZE = (9, 6)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = []
imgpointsL, imgpointsR = [], []

images_left = glob.glob("left/*.jpg")
images_right = glob.glob("right/*.jpg")

for imgL, imgR in zip(images_left, images_right):
    grayL = cv2.cvtColor(cv2.imread(imgL), cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(cv2.imread(imgR), cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

_, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC
)

np.savez("stereo_params.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print("✅ 相机校准完成，参数已保存")
