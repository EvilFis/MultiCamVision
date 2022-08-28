import os
import cv2
import numpy as np


class Calibrate:

    def __init__(self, board_size: tuple[int, int] = (6, 9),
                 world_scaling: float = 1.,
                 criteria: tuple[int, int, float] = (3, 30, 0.001)):

        """
        :param board_size: Размер шахматной доски (h, w)
        :param world_scaling: масштаб увеличения камеры
        :param criteria: критерии по поиску точек
        """

        self._board_size = board_size
        self._criteria = criteria

        self._objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
        self._objp[0, :, :2] = np.transpose(np.mgrid[0:board_size[0], 0:board_size[1]]).reshape(-1, 2)
        self._objp = world_scaling * self._objp

    def get_points_findChessboardCorners(self,
                                         images: list,
                                         save_path: str = '',
                                         win_size: tuple = (11, 11),
                                         zero_zone: tuple = (-1, -1),
                                         corners_inp: np.ndarray = None,
                                         flags: int = None) -> tuple[list, list, list]:
        """
        Поиск углов шахматной доски

        :param images: list пикселей каждой картинки
        :param save_path: Путь сохранения найденных углов на шахматной доске
        :param win_size: Половина длины стороны окна поиска.
        :param zero_zone: Половина размера мертвой зоны в середине зоны поиска, по которой суммирование не производится.
            Значение (-1,-1) указывает на то, что такого размера нет.
        :param corners_inp: Исходные координаты углов
        :param flags: Различные флаги, которые могут быть нулевыми или комбинацией значений (https://clck.ru/vK2E2)

        :return: Список удачных нахождений углов (ret_array),
            Список объектов шахматной доски (obj_points)
            Список точек шахматной доски на изображениие (img_points)
        """


        ret_array = []
        img_points = []
        obj_points = []

        for frame, val in zip(images, range(1, len(images) + 1)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(image=gray,
                                                     patternSize=(self._board_size[0], self._board_size[1]),
                                                     corners=corners_inp,
                                                     flags=flags)

            if ret:
                corners = cv2.cornerSubPix(image=gray,
                                           corners=corners,
                                           winSize=win_size,
                                           zeroZone=zero_zone,
                                           criteria=self._criteria)

                if save_path:
                    cv2.drawChessboardCorners(image=frame,
                                              patternSize=(self._board_size[0], self._board_size[1]),
                                              corners=corners,
                                              patternWasFound=ret)

                    cv2.imwrite(f"{save_path}/ChessboardCorners_{val}.jpg", frame)

                ret_array.append(ret)
                obj_points.append(self._objp)
                img_points.append(corners)

        return ret_array, obj_points, img_points

    def camera_calibrate(self, images_folder: str = './img',
                         save_path: str = '',
                         win_size: tuple = (11, 11),
                         zero_zone: tuple = (-1, -1),
                         corners_inp: np.ndarray = None,
                         flags: int = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Калибровка одной камеры

        :param images_folder: Путь до калибровочных картинок
        :param save_path: Путь сохранения найденных углов на шахматной доске
        :param win_size: Половина длины стороны окна поиска.
        :param zero_zone: Половина размера мертвой зоны в середине зоны поиска, по которой суммирование не производится.
            Значение (-1,-1) указывает на то, что такого размера нет.
        :param corners_inp: Исходные координаты углов
        :param flags: Различные флаги, которые могут быть нулевыми или комбинацией значений (https://clck.ru/vK2E2)

        :return: tuple откалиброванной камеры: Матрица камеры (mtx),
            Выходной вектор коэффициентов искажения (dist),
            Выходной вектор векторов вращения, оцененный для каждого вида шаблона (rvecs)
            Выходной вектор векторов переноса, оцененных для каждого вида шаблона (tvecs)
        """

        images = self.load_images(images_folder)

        width = images[0].shape[1]
        height = images[0].shape[0]

        _, obj_points, img_points = self.get_points_findChessboardCorners(images=images,
                                                                          save_path=save_path,
                                                                          win_size=win_size,
                                                                          zero_zone=zero_zone,
                                                                          corners_inp=corners_inp,
                                                                          flags=flags)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=obj_points,
                                                           imagePoints=img_points,
                                                           imageSize=(width, height),
                                                           cameraMatrix=None,
                                                           distCoeffs=None)

        return mtx, dist, rvecs, tvecs

    def stereo_camera_calibrate(self,
                                images_folder1: str = './img',
                                images_folder2: str = './img',
                                win_size: tuple = (11, 11),
                                zero_zone: tuple = (-1, -1),
                                corners_inp: np.ndarray = None,
                                flags: int = cv2.CALIB_FIX_INTRINSIC
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, np.ndarray]:

        """
        Производит стерео калибровку камер

        :param images_folder1: Путь до калибровочных картинок камеры 1
        :param images_folder2: Путь до калибровочных картинок камеры 2
        :param win_size: Половина длины стороны окна поиска.
        :param zero_zone: Половина размера мертвой зоны в середине зоны поиска, по которой суммирование не производится.
            Значение (-1,-1) указывает на то, что такого размера нет.
        :param corners_inp: Исходные координаты углов
        :param flags: Различные флаги, которые могут быть нулевыми или комбинацией значений (https://clck.ru/vJvii)

        :return: tuple полученных данных калибровки матрицы камеры 1 и 2 (mtx1, mtx2),
            вектор коэффициентов искажения камеры 1 и 2 (dist1, dist2),
            Матрица вращения (R) и вектор переноса (T)
        """

        images1 = self.load_images(images_folder1)
        images2 = self.load_images(images_folder2)

        mtx1, dist1, _, _ = self.camera_calibrate(images_folder=images_folder1,
                                                  win_size=win_size,
                                                  zero_zone=zero_zone,
                                                  corners_inp=corners_inp)

        mtx2, dist2, _, _ = self.camera_calibrate(images_folder=images_folder2,
                                                  win_size=win_size,
                                                  zero_zone=zero_zone,
                                                  corners_inp=corners_inp)

        width = images1[0].shape[1]
        height = images1[0].shape[0]

        img_points1 = []
        img_points2 = []

        obj_points = []

        for frame1, frame2 in zip(images1, images2):

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            ret1, corners1 = cv2.findChessboardCorners(gray1, self._board_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, self._board_size, None)

            if ret1 and ret2:
                corners1 = cv2.cornerSubPix(gray1, corners1, win_size, zero_zone, self._criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, win_size, zero_zone, self._criteria)

                obj_points.append(self._objp)
                img_points1.append(corners1)
                img_points2.append(corners2)

        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objectPoints=obj_points,
                                                                      imagePoints1=img_points1,
                                                                      imagePoints2=img_points2,
                                                                      cameraMatrix1=mtx1,
                                                                      distCoeffs1=dist1,
                                                                      cameraMatrix2=mtx2,
                                                                      distCoeffs2=dist2,
                                                                      imageSize=(width, height),
                                                                      criteria=self._criteria,
                                                                      flags=flags)

        return mtx1, dist1, mtx2, dist2, R, T

    @staticmethod
    def load_images(images_folder: str) -> list:
        """
        Считывает все изображения с указанной папки, формат поддерживаемых расширений файлов:
        `.jpg`, `.jpeg`, `.png`

        :param images_folder: Путь до изображений

        :return: Список в котором содержатся считанные пиксели изображений
        """

        images_names = [f"{images_folder}/{img_name}" for img_name in os.listdir(images_folder)
                        if img_name.endswith(('png', 'jpeg', 'jpg'))]
        images = [cv2.imread(img_name, 1) for img_name in sorted(images_names)]

        return images
