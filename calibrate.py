import os
import cv2

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

mtx, dist, rvecs, tvecs = camera_calibrate(boar_size=(6, 9),
                                           criteria=criteria,
                                           images_path=f"./{PATH_IMG}/camera 1/",
                                           draw_chessboard=False)

mtx2, dist2, rvecs2, tvecs2 = camera_calibrate(boar_size=(6, 9),
                                               criteria=criteria,
                                               images_path=f"./{PATH_IMG}/camera 2/",
                                               draw_chessboard=False)

# Значения по диагонали - фокусное расстояние `X` и `Y`
# Последний столбец - координаты `X` и `Y` оптического центра в плоскости изображения
print("Matrix camera 1:\n", mtx, end="\n\n")
print("Matrix camera 2:\n", mtx2)


# ================================================ TEST ================================================================
# _, rvecs2, tvecs2, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
# axis = np.float32([0, 0, 0])
# imgpts, jac = cv2.projectPoints(axis, rvecs2, tvecs2, mtx, dist)
