import os
import cv2
import numpy as np

from dotenv import load_dotenv

from cam import Camera, camera_calibrate, create_dataset

load_dotenv()

PATH_IMG = './camera_img'

# Загрузка данных о камере
path1 = os.getenv("RTSP_CAMERA_MAIN")
path2 = os.getenv("RTSP_CAMERA_ADDED")

# =========================================== СОЗДАНИЕ КАРТИНОК ========================================================
camera1 = Camera(camera_id=0, show_frame=False, vertical_flip=True, save_video=False)
camera2 = Camera(camera_id=1, show_frame=False, vertical_flip=True, save_video=False)
# camera3 = Camera(camera_id=path1, show_frame=False, vertical_flip=True, save_video=False)
# camera4 = Camera(camera_id=path2, show_frame=False, vertical_flip=True, save_video=False)

# camera1.initialize()
# camera2.initialize()
# camera3.initialize()
# camera4.initialize()

# create_dataset([camera1, camera2], dir=PATH_IMG, zoom=.8)

# ====================================== КАЛИБРОВКА КАМЕРЫ =============================================================
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# mtx, dist, rvecs, tvecs = camera_calibrate(boar_size=(6, 9),
#                                            criteria=criteria,
#                                            images_path=f"./{PATH_IMG}/camera 1/",
#                                            draw_chessboard=False)
#
mtx2, dist2, rvecs2, tvecs2, obj_points, img_points = camera_calibrate(boar_size=(6, 9),
                                                                       criteria=criteria,
                                                                       images_path=f"./{PATH_IMG}/camera 2/",
                                                                       draw_chessboard=False)

# Значения по диагонали - фокусное расстояние `X` и `Y`
# Последний столбец - координаты `X` и `Y` оптического центра в плоскости изображения
# print("Matrix camera 1:\n", mtx, end="\n\n")
# print("Matrix camera 2:\n", mtx2, end="\n\n\n\n")
#
# print("Dist camera 1:\n", dist, end="\n\n")
# print("Dist camera 2:\n", dist2, end="\n\n\n\n")
#
# print("Rvecs camera 1:\n", rvecs, end="\n\n")
# print("Rvecs camera 2:\n", rvecs2, end="\n\n\n\n")
#
# print("Tvecs camera 1:\n", tvecs, end="\n\n")
# print("Tvecs camera 2:\n", tvecs2, end="\n\n\n\n")


# ================================================ TEST ================================================================
# _, rvecs2, tvecs2, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
# axis = np.float32([0, 0, 0])
# imgpts, jac = cv2.projectPoints(axis, rvecs2, tvecs2, mtx, dist)

print("Calibration done")


def groundProjectPoint(image_point, camera_matrix, rotMat, tvec, z=0.0):
    camMat = np.asarray(camera_matrix)
    iRot = np.linalg.inv(rotMat)
    iCam = np.linalg.inv(camMat)

    uvPoint = np.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return wcPoint


def mousePoint(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global x_global, y_global, z_global, x_image, y_image
        print(f"Отправил координаты изображения: {x}x{y}")
        x_image = x
        y_image = y

        x_global, y_global, z_global = groundProjectPoint((x, y), mtx2, rotMat, tvecs2[-1])
        # print(f"Получил мировые координаты: {x[0]}x{y[0]}x{z[0]}")
        imgpts, jac = cv2.projectPoints((x_global[0], y_global[0], z_global[0]), rvecs2[-1], tvecs2[-1], mtx2, dist2)
        print(f"Получил координаты изображения: {imgpts[0]}")


camera2.initialize()

# print(f"Obj points -1 = {obj_points[-1][0].shape}")
# print(f"Img points -1 = {np.asarray(img_points[-1])}")
#
# print(f"Obj points 0 = {obj_points[0]}")
# print(f"Img points 0 = {img_points[0]}")

rotMat, _ = cv2.Rodrigues(rvecs2[-1])

x_image = 0
y_image = 0
x_global = 0
y_global = 0
z_global = 0

while True:
    frame = camera2.read_frame()

    if x_global and x_image:
        cv2.circle(frame, (x_image, y_image), 5, (0, 255, 0), cv2.FILLED)
        cv2.putText(frame,
                    f"Image cords {x_image}x{y_image}",
                    (x_image, (y_image - 15)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=1)
        cv2.putText(frame,
                    f"Global cords {round(x_global[0], 2)}x{round(y_global[0], 2)}x{round(z_global[0], 2)}",
                    (x_image, (y_image + 15)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=1)

    cv2.imshow('Camera', frame)
    cv2.setMouseCallback('Camera', mousePoint, param=[frame])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera2.release()
        cv2.destroyAllWindows()
        break
