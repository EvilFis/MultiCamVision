import sys
import cv2
import time
import numpy as np

from threading import Thread, Event


class Camera(object):

    def __init__(self, camera_id: int | float | str = 0,
                 show_frame: bool = False,
                 vertical_flip: bool = False,
                 settings: dict = None):

        if sys.platform == 'linux' or sys.platform == 'linux2':
            if isinstance(camera_id, int):
                camera_id = f"/dev/video{camera_id}"

        self._camera_id = camera_id

        if not isinstance(vertical_flip, bool) and vertical_flip:
            self._vertical_flip = True
        else:
            self._vertical_flip = False

        self._settings = settings
        self._stop = False
        # self._t0 = time.time()

        self._thread_ready = Event()
        self._thread = Thread(name="Update frame", target=self._update_frame, args=(show_frame,))

    def initialize(self, show_frame: bool = False):
        self.test_camera()
        self._stop = False

        self.start_camera()
        self._thread_ready.wait()

    def start_camera(self):
        self._thread.start()
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

    def _update_frame(self, show_frame: bool = False):
        while not self._stop:
            ret, frame = self._cap.read()

            if not ret:
                print(f"Cam {self._camera_id} | Error reading frame!")

            if self._vertical_flip:
                frame = cv2.flip(frame, -1)

            if show_frame:
                cv2.imshow(f'Camera {self._camera_id}', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
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
                        img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                     None, scale, scale)
                    if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
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

