import os
import pickle
import numpy as np


def save_pickle_data(data, name: str = "", path: str = './') -> None:
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f"{path}/{name}.pickle", 'wb') as handle:
        pickle.dump(data, handle)

    print(f"[#] Save data in {path}/{name}.pickle completed successfully")


def load_pickle_data(name: str, path: str = './'):
    with open(f"{path}/{name}.pickle", 'rb') as handle:
        data = pickle.load(handle)

    return data


def get_projection_matrix(camera_matrix1, camera_matrix2, R, T):
    P1 = camera_matrix1 @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P2 = camera_matrix2 @ np.concatenate([R, T], axis=-1)

    return P1, P2


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]]

    A = np.array(A).reshape((4, 4))
    B = np.dot(A.transpose(), A)

    U, s, Vh = np.linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def get_3d_points(key_points1, key_points2, P1, P2):
    p3ds = []
    for uv1, uv2 in zip(key_points1, key_points2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    return np.array(p3ds)


def visualization_3d(key_points1, key_points2,
                     matrix1, matrix2, R, T,
                     **kwargs):
    if not isinstance(key_points1[0][0], list) and not isinstance(key_points2[0][0], list):
        key_points1 = [[kp for kp in key_points1]]
        key_points2 = [[kp for kp in key_points2]]

    if len(key_points1) != len(key_points2):
        raise ValueError('Разная длинна кадров')

    P1, P2 = get_projection_matrix(matrix1, matrix2, R, T)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(-90, -90)

    for kps_f1, kps_f2 in zip(key_points1, key_points2):
        p3ds = get_3d_points(kps_f1, kps_f2, P1, P2)

        if 'connections' in kwargs and kwargs['connections']:
            for _c in kwargs['connections']:
                ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]],
                        ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
                        zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]],
                        linewidth=4,
                        color='green')

                if 'show_points' in kwargs and kwargs['show_points']:
                    ax.scatter(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]],
                               ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
                               zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]],
                               linewidth=4,
                               color='red')
        else:
            ax.scatter(xs=p3ds[:, 0],
                       ys=p3ds[:, 1],
                       zs=p3ds[:, 2],
                       c='red')

        if 'showAxis' in kwargs and not kwargs['showAxis']:
            ax.set_axis_off()

        if len(key_points1) == 1:
            plt.show()

        else:
            plt.draw()
            plt.pause(.001)
            ax.clear()