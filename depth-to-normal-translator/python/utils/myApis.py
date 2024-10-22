import cv2
import numpy as np

kernel_Gx = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]])

kernel_Gy = np.array([[0, -1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])

cp2tv_Gx = np.array([[0, 0, 0],
                     [0, -1, 1],
                     [0, 0, 0]])

cp2tv_Gy = np.array([[0, 0, 0],
                     [0, -1, 0],
                     [0, 1, 0]])

lap_ker_alpha = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

lap_ker_beta = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])

lap_ker_gamma = np.array([[0.25, 0.5, 0.25],
                          [0.5, -3, 0.5],
                          [0.25, 0.5, 0.25]])

gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1],
                       [1],
                       [0]])
gradient_d = np.array([[0],
                       [-1],
                       [1]])

laplace_hor = np.array([[-1, 2, -1]])

laplace_ver = np.array([[-1],
                        [2],
                        [-1]])


def soft_min(laplace_map, base, direction):
    """

    :param laplace_map: the horizontal laplace map or vertical laplace map, shape = [vMax, uMax]
    :param base: the base of the exponent operation
    :param direction: 0 for horizontal, 1 for vertical
    :return: weighted map (lambda 1,2 or 3,4)
    """
    h, w = laplace_map.shape
    eps = 1e-8  # to avoid division by zero

    lap_power = np.power(base, -laplace_map)
    if direction == 0:  # horizontal
        lap_pow_l = np.hstack([np.zeros((h, 1)), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((h, 1))])
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), \
               (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r)

    elif direction == 1:  # vertical
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), \
               (lap_pow_d + eps / 2) / (eps + lap_pow_u + lap_pow_d)


def get_filter(Z, cp2tv=False):
    """get partial u, partial v"""
    if cp2tv:
        Gu = cv2.filter2D(Z, -1, cp2tv_Gx)
        Gv = cv2.filter2D(Z, -1, cp2tv_Gy)
    else:
        Gu = cv2.filter2D(Z, -1, kernel_Gx) / 2
        Gv = cv2.filter2D(Z, -1, kernel_Gy) / 2
    return Gu, Gv


def get_DAG_filter(Z, base=np.e, lap_conf='1D-DLF'):
    # calculate gradients along four directions
    grad_l = cv2.filter2D(Z, -1, gradient_l)
    grad_r = cv2.filter2D(Z, -1, gradient_r)
    grad_u = cv2.filter2D(Z, -1, gradient_u)
    grad_d = cv2.filter2D(Z, -1, gradient_d)

    # calculate laplace along 2 directions
    if lap_conf == '1D-DLF':
        lap_hor = abs(grad_l - grad_r)
        lap_ver = abs(grad_u - grad_d)
    elif lap_conf == 'DLF-alpha':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_alpha))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_alpha))
    elif lap_conf == 'DLF-beta':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_beta))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_beta))
    elif lap_conf == 'DLF-gamma':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_gamma))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_gamma))
    else:
        raise ValueError

    lambda_map1, lambda_map2 = soft_min(lap_hor, base, 0)
    lambda_map3, lambda_map4 = soft_min(lap_ver, base, 1)

    eps = 1e-8
    thresh = base
    lambda_map1[lambda_map1 / (lambda_map2 + eps) > thresh] = 1
    lambda_map2[lambda_map1 / (lambda_map2 + eps) > thresh] = 0
    lambda_map1[lambda_map2 / (lambda_map1 + eps) > thresh] = 0
    lambda_map2[lambda_map2 / (lambda_map1 + eps) > thresh] = 1

    lambda_map3[lambda_map3 / (lambda_map4 + eps) > thresh] = 1
    lambda_map4[lambda_map3 / (lambda_map4 + eps) > thresh] = 0
    lambda_map3[lambda_map4 / (lambda_map3 + eps) > thresh] = 0
    lambda_map4[lambda_map4 / (lambda_map3 + eps) > thresh] = 1

    # lambda_maps = [lambda_map1, lambda_map2, lambda_map3, lambda_map4]
    Gu = lambda_map1 * grad_l + lambda_map2 * grad_r
    Gv = lambda_map3 * grad_u + lambda_map4 * grad_d
    return Gu, Gv


def MRF_optim(depth, n_est, lap_conf='DLF-alpha'):
    h, w = depth.shape
    n_x, n_y, n_z = n_est[:, :, 0], n_est[:, :, 1], n_est[:, :, 2]
    # =====================optimize the normal with MRF=============================
    if lap_conf == '1D-DLF':
        Z_laplace_hor = abs(cv2.filter2D(depth, -1, laplace_hor))
        Z_laplace_ver = abs(cv2.filter2D(depth, -1, laplace_ver))

        # [x-1,y] [x+1,y] [x,y-1] [x,y+1], [x,y]
        Z_laplace_stack = np.array((np.hstack((np.inf * np.ones((h, 1)), Z_laplace_hor[:, :-1])),
                                    np.hstack((Z_laplace_hor[:, 1:], np.inf * np.ones((h, 1)))),
                                    np.vstack((np.inf * np.ones((1, w)), Z_laplace_ver[:-1, :])),
                                    np.vstack((Z_laplace_ver[1:, :], np.inf * np.ones((1, w)))),
                                    (Z_laplace_hor + Z_laplace_ver) / 2))
    else:
        if lap_conf == 'DLF-alpha':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_alpha))
        elif lap_conf == 'DLF-beta':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_beta))
        elif lap_conf == 'DLF-gamma':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_gamma))
        else:
            raise ValueError
        Z_laplace_stack = np.array((np.hstack((np.inf * np.ones((h, 1)), Z_laplace[:, :-1])),
                                    np.hstack((Z_laplace[:, 1:], np.inf * np.ones((h, 1)))),
                                    np.vstack((np.inf * np.ones((1, w)), Z_laplace[:-1, :])),
                                    np.vstack((Z_laplace[1:, :], np.inf * np.ones((1, w)))),
                                    Z_laplace))

    # best_loc_map: 0 for left, 1 for right, 2 for up, 3 for down, 4 for self
    best_loc_map = np.argmin(Z_laplace_stack, axis=0)
    Nx_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_x[:, :-1])),
                           np.hstack((n_x[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_x[:-1, :])),
                           np.vstack((n_x[1:, :], np.zeros((1, w)))),
                           n_x)).reshape(5, -1)
    Ny_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_y[:, :-1])),
                           np.hstack((n_y[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_y[:-1, :])),
                           np.vstack((n_y[1:, :], np.zeros((1, w)))),
                           n_y)).reshape(5, -1)
    Nz_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_z[:, :-1])),
                           np.hstack((n_z[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_z[:-1, :])),
                           np.vstack((n_z[1:, :], np.zeros((1, w)))),
                           n_z)).reshape(5, -1)

    n_x = Nx_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_y = Ny_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_z = Nz_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_est = cv2.merge((n_x, n_y, n_z))
    return n_est
