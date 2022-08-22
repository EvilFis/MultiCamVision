import os
import sys
import cv2
import time
import shutil
import numpy as np

from threading import Thread, Event


class Camera(object):

    def __init__(self, camera_id: int | float | str = 0,
                 show_frame: bool = False,
                 vertical_flip: bool = False,
                 save_video: bool = False,
                 settings: dict = None):

        if sys.platform == 'linux' or sys.platform == 'linux2':
            if isinstance(camera_id, int):
                camera_id = f"/dev/video{camera_id}"

        self._camera_id = camera_id

        if not isinstance(vertical_flip, bool) and vertical_flip:
            self._vertical_flip = True
        else:
            self._vertical_flip = False

        if not isinstance(save_video, bool) and save_video:
            self._save_video = True
        else:
            self._save_video = False

        self._settings = settings
        self._stop = False
        # self._t0 = time.time()

        self._thread_ready = Event()
        self._thread = Thread(name="Update frame", target=self._update_frame, args=(show_frame,))

    # def __del__(self):
    #     self.release()

    def initialize(self):
        self.test_camera()
        self._stop = False

        self._start_camera()
        self._thread_ready.wait()

    def read_frame(self):
        return self._current_frame

    def test_camera(self, ):

        self._setup()

        ret, test_frame = self._cap.read()
        warn = 0

        if not ret:
            raise ValueError(f"The camera `{self._camera_id}` could not read the image")

        if test_frame.shape[0] != self._height:
            print("WARNING: Camera height is different from the setting one!")
            warn += 1

        if test_frame.shape[1] != self._weight:
            raise ValueError("Camera width is different from the setting one!")

        print(f"Camera {self._camera_id} testing completed successfully. Warnings: {warn}")

    def set_camera_setting(self):
        pass

    def release(self):
        self._stop = True
        time.sleep(0.1)
        self._cap.release()

    def _start_camera(self):
        self._thread.start()
        self._thread_ready.wait()

    def _update_frame(self, show_frame: bool = False):

        _, frame_test = self._cap.read()
        out_file = cv2.VideoWriter(f"Camera {self._camera_id}",
                                   cv2.VideoWriter_fourcc(*"MP4V"),
                                   25.0,
                                   (frame_test.shape[1], frame_test.shape[0]))

        while not self._stop:
            ret, frame = self._cap.read()

            if not ret:
                print(f"Cam {self._camera_id} | Error reading frame!")

            if self._vertical_flip:
                frame = cv2.flip(frame, -1)

            if self._save_video:
                out_file.write(frame)

            if show_frame:

                key = cv2.waitKey(1) & 0xFF
                cv2.imshow(f'Camera {self._camera_id}', frame)

                if key == ord('q'):
                    self.release()

            self._current_frame = frame
            self._thread_ready.set()

    def _setup(self):

        if sys.platform == 'linux' or sys.platform == 'linux2':
            self._cap = cv2.VideoCapture(self._camera_id, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(self._camera_id)

        if not self._cap.isOpened():
            raise ValueError(f"Camera `{self._camera_id}` is not found!")

        self.set_camera_setting()

        self._weight = int(self._cap.get(3))
        self._height = int(self._cap.get(4))

    @staticmethod
    def stack_images(scale: int | float, img_array: list | tuple):
        rows = len(img_array)
        cols = len(img_array[0])
        rows_available = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rows_available:
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y],
                                                     (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                     None, scale, scale)
                    if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y],
                                                                                       cv2.COLOR_GRAY2BGR)
            image_blank = np.zeros((height, width, 3), np.uint8)
            hor = [image_blank] * rows
            hor_con = [image_blank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                              scale)
                if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(img_array)
            ver = hor
        return ver


def camera_calibrate(boar_size: tuple = (6, 9),
                     criteria: tuple = None,
                     images_path: str = '',
                     draw_chessboard: bool = True) -> tuple:
    """
    :param boar_size:
    :param criteria:
    :param images_path:
    :param draw_chessboard:
    :return: mtx - Внутренняя матрица камеры
    dist - Коэффициенты дисторсии объектива
    rvecs - Вращение задано как вектор 3×1. Направление вектора задает ось вращения, а величина вектора задает угол поворота.
    tvecs - Вектор перевода 3×1.
    """

    images_list = [images_path + img for img in os.listdir(images_path)
                   if img.endswith(".jpg")]
    gray = None

    obj_points = []
    img_points = []

    objp = np.zeros((1, boar_size[0] * boar_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:boar_size[0], 0:boar_size[1]].T.reshape(-1, 2)

    # Находим точки шахматной доски
    for fname in images_list:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(image=gray,
                                                 patternSize=boar_size,
                                                 corners=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not ret:
            print(f"[#] Image {fname} doesn't open")
            continue

        obj_points.append(objp)

        corner_sub_pix = cv2.cornerSubPix(image=gray,
                                          corners=corners,
                                          winSize=(11, 11),
                                          zeroZone=(-1, -1),
                                          criteria=criteria)

        img_points.append(corner_sub_pix)

        # Отображение шахматной доски
        if draw_chessboard:
            img = cv2.drawChessboardCorners(image=img,
                                            patternSize=boar_size,
                                            corners=corners,
                                            patternWasFound=ret)

            cv2.imshow('Draw chessboard corners', img)
            cv2.waitKey(0)

    try:
        cv2.destroyWindow("Draw chessboard corners")
    except cv2.error:
        pass

    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=obj_points,
                                                     imagePoints=img_points,
                                                     imageSize=gray.shape[::-1],
                                                     cameraMatrix=None,
                                                     distCoeffs=None)

    return mtx, dist, rvecs, tvecs


def create_dataset(camera_list: list,
                   dir: str = "./camera_img",
                   zoom: float = 0.5) -> None:

    # Проверка структуры папок
    if os.path.isdir(dir):
        shutil.rmtree(dir)

    # Создание директории
    for i in range(1, len(camera_list) + 1):
        os.makedirs(f"./{dir}/camera {i}")

    # Черное изображение
    black_img = np.zeros((1024, 1024, 3), np.uint8)

    # Подгоняем длину камеры до трех
    cam_len = len(camera_list)
    while True:
        if cam_len % 3 == 0:
            break

        cam_len += 1

    counter = 0
    while True:

        frames = []
        frames_struct = []

        # Перебираем все камеры и записываем кадры в список
        for cam_id in range(cam_len + 1):

            if cam_id % 3 == 0 and cam_id != 0:
                frames_struct.append(frames.copy())
                frames.clear()

            if cam_id == cam_len: break

            try:
                frames.append(camera_list[cam_id].read_frame())

            except IndexError:
                frames.append(black_img)

        # Объединение всех кадров
        frame = Camera.stack_images(zoom, frames_struct)

        cv2.imshow("Create dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        # Скриншот
        if key == ord('s') or key == ord('S'):

            counter += 1

            for cam_id in range(len(camera_list)):
                cv2.imwrite(os.path.join(f"{dir}/camera {cam_id + 1}",
                                         f"Camera {cam_id + 1}_{counter}.jpg"),
                            camera_list[cam_id].read_frame())

            print(f'Screenshot {counter} is made')

        # Выход с программы
        if key == ord('q') or key == ord('Q'):
            cv2.destroyWindow("Create dataset")

            for cam in camera_list:
                cam.release()

            break

    print("[!] Dataset creates")


if __name__ == "__main__":
    os.chdir("..")

    camera1 = Camera(camera_id=0, show_frame=False, vertical_flip=True, save_video=False)

    camera1.initialize()

    create_dataset([camera1])
