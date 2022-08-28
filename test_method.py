import os
import shutil
import pickle
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from cam import create_dataset, Camera


def save_data(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)

    print("Saved")


def load_data(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def camera_calibrate(images_folder='./img',
                     board_size=(6, 9),
                     world_scaling=1.,
                     debug=False):
    images_names = sorted(glob.glob(images_folder))
    images = []

    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)

    # критерии, используемые детектором шахматной доски.
    # Измените это, если код не может найти шахматную доску
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = images[0].shape[1]
    height = images[0].shape[0]

    imgpoints = []
    objpoints = []

    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (board_size[0], board_size[1]), None)

        if ret:

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (board_size[0], board_size[1]), corners, ret)

            if debug:
                cv2.imshow('img', frame)
                k = cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    return mtx, dist


def stereo_camera_calibrate(images_folder1='./img',
                            images_folder2='./img',
                            board_size=(6, 9),
                            world_scaling=1.,
                            cameraMatrix1=None,
                            distCoeffs1=None,
                            cameraMatrix2=None,
                            distCoeffs2=None,
                            debug=False):
    cam1_path = sorted(glob.glob(images_folder1))
    cam2_path = sorted(glob.glob(images_folder2))

    c1_images = []
    c2_images = []

    for im1, im2 in zip(cam1_path, cam2_path):
        im = cv2.imread(im1, 1)
        c1_images.append(im)

        im = cv2.imread(im2, 1)
        c2_images.append(im)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []

    objpoints = []

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, board_size, None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, board_size, None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            if debug:
                cv2.drawChessboardCorners(frame1, board_size, corners1, c_ret1)
                cv2.imshow('img', frame1)

                cv2.drawChessboardCorners(frame2, board_size, corners2, c_ret2)

                cv2.imshow('img2', frame2)
                cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                                  imagePoints1=imgpoints_left,
                                                                  imagePoints2=imgpoints_right,
                                                                  cameraMatrix1=cameraMatrix1,
                                                                  distCoeffs1=distCoeffs1,
                                                                  cameraMatrix2=cameraMatrix2,
                                                                  distCoeffs2=distCoeffs2,
                                                                  imageSize=(width, height),
                                                                  criteria=criteria,
                                                                  flags=stereocalibration_flags)

    return R, T


def mousePoint(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}]")


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    # from scipy import linalg
    U, s, Vh = np.linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3, 0:3] / Vh[3, 3])

    return Vh[3, 0:3] / Vh[3, 3]


if __name__ == "__main__":
    camera1 = Camera(camera_id=0, show_frame=False, vertical_flip=True, save_video=False)
    camera2 = Camera(camera_id=1, show_frame=False, vertical_flip=True, save_video=False)
    #
    # create_screen(0)
    # create_screen(1)
    #
    # camera1.initialize()
    # camera2.initialize()

    # create_dataset([camera1, camera2], './img/split/')

    # ========================================== КАЛИБРОВКА КАМЕРЫ =====================================================
    # mtx1, dist1 = camera_calibrate('./img/split/camera 1/*.jpg', debug=False)
    # mtx2, dist2 = camera_calibrate('./img/split/camera 2/*.jpg', debug=False)
    #
    # R, T = stereo_camera_calibrate(images_folder1="./img/split/camera 1/*.jpg",
    #                                images_folder2="./img/split/camera 2/*.jpg",
    #                                cameraMatrix1=mtx1,
    #                                cameraMatrix2=mtx2,
    #                                distCoeffs1=dist1,
    #                                distCoeffs2=dist2,
    #                                debug=False)

    # ==================================== СОХРАНЕНИЕ ДАННЫХ ===========================================================
    # save_data('./data/matrix_camera_1080.pickle', mtx1)
    # save_data('./data/matrix_camera.pickle', mtx2)
    #
    # save_data('./data/dist_camera_1080.pickle', dist1)
    # save_data('./data/dist_camera.pickle', dist2)
    #
    # save_data('./data/stereo_R.pickle', R)
    # save_data('./data/stereo_T.pickle', T)

    # ============================================== ЗАГРУЗКА ДАННЫХ ===================================================
    mtx2 = load_data('./data/matrix_camera_1080.pickle')
    mtx1 = load_data('./data/matrix_camera.pickle')

    dist2 = load_data('./data/dist_camera_1080.pickle')
    dist1 = load_data('./data/dist_camera.pickle')

    R = load_data('./data/stereo_R.pickle')
    T = load_data('./data/stereo_T.pickle')

    print(f"Camera matrix 0:\n {mtx1}")
    print(f"Camera matrix 1:\n {mtx2}")

    print(f"Camera dist 0:\n {dist1}")
    print(f"Camera dist 1:\n {dist2}")

    print(f"R:\n {R}")
    print(f"T:\n {T}")

    # board_size = (6, 9)
    # world_scaling = 1.

    # =============================================== РУЧНАЯ РАЗМЕТКА ДАННЫХ ===========================================
    # count = 0
    # while True:
    #
    #     if not count:
    #         path = './img/1.jpg'
    #     else:
    #         path = './img/2.jpg'
    #
    #     img = cv2.imread(path, 1)
    #
    #     cv2.imshow("Img", img)
    #     cv2.setMouseCallback('Img', mousePoint)
    #
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         if not count:
    #             count += 1
    #             continue
    #
    #         cv2.destroyAllWindows()
    #         break
    #                   # Право     #Середина  # лево
    # uvs1 = np.array([[249, 175], [187, 177], [106, 166],
    #                  [67, 296], [163, 409], [257, 289],
    #                  [267, 408], [190, 405]])
    #
    # uvs2 = np.array([[506, 50], [408, 52], [321, 53],
    #                  [286, 196], [355, 320], [503, 189],
    #                  [494, 329], [398, 321]])

    # frame1 = cv2.imread('./img/1.jpg')
    # frame2 = cv2.imread('./img/2.jpg')
    #
    # plt.imshow(frame1[:, :, [2, 1, 0]])
    # plt.scatter(uvs1[:, 0], uvs1[:, 1])
    # plt.show()
    #
    # plt.imshow(frame2[:, :, [2, 1, 0]])
    # plt.scatter(uvs2[:, 0], uvs2[:, 1])
    # plt.show()
    # #
    # RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    # P1 = mtx1 @ RT1
    #
    # RT2 = np.concatenate([R, T], axis=-1)
    # P2 = mtx2 @ RT2
    #
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(0, -40)
    # ax.set_ylim3d(-20, 20)
    # ax.set_zlim3d(50, 100)
    #
    # p3ds = []
    # for uv1, uv2 in zip(uvs1, uvs2):
    #     _p3d = DLT(P1, P2, uv1, uv2)
    #     p3ds.append(_p3d)
    # p3ds = np.array(p3ds)
    #
    # connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [1, 7]]
    # for _c in connections:
    #     # print(p3ds[_c[0]])
    #     # print(p3ds[_c[1]])
    #     ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
    #             zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
    #
    # plt.show()

    # =============================================== НАХОЖДЕНИЕ АВТО =================================================

    import mediapipe as mp
    from mpl_toolkits.mplot3d import Axes3D


    def get_frame_keypoints(landmarks, frame):
        frame_keypoints = []
        print(landmarks)
        for face_landmarks in landmarks:
            for p in range(21):
                pxl_x = int(round(frame.shape[1] * face_landmarks.landmark[p].x))
                pxl_y = int(round(frame.shape[0] * face_landmarks.landmark[p].y))
                kpts = [pxl_x, pxl_y]
                frame_keypoints.append(kpts)

        return frame_keypoints


    mp_drawing = mp.solutions.drawing_utils
    # mp_face = mp.solutions.face_mesh
    mp_face = mp.solutions.hands

    # face1 = mp_face.FaceMesh(max_num_faces=1,
    #                          refine_landmarks=True,
    #                          min_detection_confidence=0.5,
    #                          min_tracking_confidence=0.5)
    # face2 = mp_face.FaceMesh(max_num_faces=1,
    #                          refine_landmarks=True,
    #                          min_detection_confidence=0.5,
    #                          min_tracking_confidence=0.5)

    face1 = mp_face.Hands(max_num_hands=1,
                          model_complexity=0,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    face2 = mp_face.Hands(max_num_hands=1,
                          model_complexity=0,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    camera1.initialize()
    camera2.initialize()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # connections = [[i, i+1] for i in range(467)]
    ax.view_init(-90, -90)

    mp_pose = mp.solutions.pose

    connections = mp_face.HAND_CONNECTIONS

    counter = 0
    global_kps1 = []
    global_kps2 = []

    while True:

        frame1 = camera1.read_frame()
        frame2 = camera2.read_frame()

        frame1_copy = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_copy = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        frame1_copy.flags.writeable = False
        frame2_copy.flags.writeable = False

        results1 = face1.process(frame1_copy)
        results2 = face2.process(frame2_copy)

        # if results1.multi_face_landmarks:
        if results1.multi_hand_landmarks:
            frame1_keypoints = get_frame_keypoints(results1.multi_hand_landmarks,
                                                   frame1)
        else:
            # frame1_keypoints = [[-1, -1]] * 468
            frame1_keypoints = [[-1, -1]] * 21

        if results2.multi_hand_landmarks:
            frame2_keypoints = get_frame_keypoints(results2.multi_hand_landmarks,
                                                   frame2)
        else:
            frame2_keypoints = [[-1, -1]] * 21

        global_kps1.append(frame1_keypoints)
        global_kps2.append(frame2_keypoints)

        # print("Frame kp 1:\n", frame1_keypoints)
        # print("Frame kp 2:\n", frame2_keypoints)

        for points1, points2 in zip(frame1_keypoints, frame2_keypoints):
            cv2.circle(frame1, points1, 1, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame2, points2, 1, (255, 0, 0), cv2.FILLED)

        frames = Camera().stack_images(0.8, [[frame1, frame2]])

        cv2.imshow('Check', frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera1.release()
            camera2.release()
            save_data('data/glob1_kps.pickle', global_kps1)
            save_data('data/glob2_kps.pickle', global_kps2)
            break

        # @ - матричное умножение
        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        P1 = mtx1 @ RT1

        RT2 = np.concatenate([R, T], axis=-1)
        P2 = mtx2 @ RT2

        p3ds = []
        for uv1, uv2 in zip(frame1_keypoints, frame2_keypoints):
            _p3d = DLT(P1, P2, uv1, uv2)
            p3ds.append(_p3d)
        p3ds = np.array(p3ds)

        for _c in connections:
            ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]],
                    ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
                    zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]],
                    c='red')
            ax.scatter(xs=[p3ds[:, 0], p3ds[:, 0]],
                       ys=[p3ds[:, 1], p3ds[:, 1]],
                       zs=[p3ds[:, 2], p3ds[:, 2]],
                       c='green')

        # ax.set_axis_off()
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        plt.draw()
        plt.pause(.001)
        ax.clear()

    # save_data('./data/glob1_kps.pickle', global_kps1)
    # save_data('./data/glob2_kps.pickle', global_kps2)

    # # ax.set_xlim3d(-14, -24)
    # # ax.set_ylim3d(-5, 5)
    # # ax.set_zlim3d(-500, 500)
    #

    #
    # connections = [[0, 1], [1, 2], [2, 3], [3, 4],
    #                [0,5], [5,6], [6,7], [7,8],
    #                [5,9], [9,10], [10,11], [11,12],
    #                [9,13], [13,14], [14,15], [15,16],
    #                [13,17], [17,18], [18,19], [19,20], [17, 0]]
    #
    # for _c in connections:
    #     # print(p3ds[_c[0]])
    #     # print(p3ds[_c[1]])
    #     ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
    #             zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
    #     ax.scatter(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
    #             zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='green')
    #
    # def animate(i):
    #     print(i/360 * 100, "%")
    #     line = ax.view_init(210, i)
    #     return line
    #
    # import matplotlib.animation as animation
    #
    # #  Создаем объект анимации:
    # sin_animation = animation.FuncAnimation(fig,
    #                                         animate,
    #                                         frames=np.linspace(0, 360, 360),
    #                                         interval = 10,
    #                                         repeat = False)
    #
    # #  Сохраняем анимацию в виде gif файла:
    # sin_animation.save('моя анимация.gif',
    #                    writer='imagemagick',
    #                    fps=30)

    # for angle in range(0, 360):
    #     ax.view_init(210, angle)
    #     plt.draw()
    #     plt.pause(.001)

# %%
