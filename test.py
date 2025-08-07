import numpy as np
import matplotlib.pyplot as plt
from numba.typed import List
from numba import njit


from config import Pathes_init
from core import KERNELS_RAYS
from probability_model_numba import pack_rays_for_numba


OBST = (Pathes_init == -1).astype(np.uint8)

dy_rays, dx_rays, w_rays = pack_rays_for_numba(KERNELS_RAYS)




# New: 

from numba import njit
import numpy as np

@njit(cache=True)
def segment_hits_obstacle(y0, x0, y1, x1, Pathes, N_y, N_x, periodic):
    dy = y1 - y0
    dx = x1 - x0
    sy = 1 if dy >= 0 else -1
    sx = 1 if dx >= 0 else -1
    dy = abs(dy)
    dx = abs(dx)

    y = y0
    x = x0

    if dx >= dy:
        err = dx // 2
        for _ in range(dx + 1):
            if 0 <= y < N_y:
                xm = x % N_x if periodic else x
                if 0 <= xm < N_x and Pathes[y, xm] == -1:
                    return True
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
    else:
        err = dy // 2
        for _ in range(dy + 1):
            if 0 <= y < N_y:
                xm = x % N_x if periodic else x
                if 0 <= xm < N_x and Pathes[y, xm] == -1:
                    return True
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
    return False




@njit(cache=True)
def path_component_with_occlusion_rays_numba(y0, x0,
                                             dirs_idx,
                                             Pathes,
                                             N_y, N_x,
                                             tau,
                                             dy_rays, dx_rays, w_rays,
                                             periodic):
    k = dirs_idx.shape[0]
    S = np.zeros(k, dtype=np.float32)
    Vision_k = Pathes.copy()

    for i in range(k):
        d_idx = dirs_idx[i]
        rays_dy = dy_rays[d_idx]
        rays_dx = dx_rays[d_idx]
        rays_w  = w_rays [d_idx]

        s_dir = 0.0
        for r in range(len(rays_dy)):
            dy = rays_dy[r]; dx = rays_dx[r]; w = rays_w[r]

            # предыдущая точка на луче (для первого шага проверим просто саму клетку)
            y_prev = y0
            x_prev = x0

            blocked = False
            for p in range(dy.shape[0]):
                yy = y0 + dy[p]
                xx = x0 + dx[p]

                # Проверяем отрезок (y_prev,x_prev) -> (yy,xx)
                if segment_hits_obstacle(y_prev, x_prev, yy, xx,
                                         Pathes, N_y, N_x, periodic):
                    blocked = True
                    break

                # накопление вклада
                if 0 <= yy < N_y:
                    xm = xx % N_x if periodic else xx
                    if 0 <= xm < N_x:
                        s_dir += Pathes[yy, xm] * w[p]
                        Vision_k[yy,xm] = 1

                y_prev = yy
                x_prev = xx

            if blocked:
                continue

        S[i] = s_dir

    if tau > 0.0:
        m = 0.0
        for i in range(k):
            if S[i] > m: m = S[i]
        if m > 0.0:
            thr = tau * m
            for i in range(k):
                if S[i] < thr: S[i] = 0.0

    return S,Vision_k










dirs_idx = np.array([7])

y0 = 50
x0 = 50

Pathes = np.zeros((100,100))

# Building:

Pathes[60:65, 30:70] = -1


N_y = 100
N_x = 100

tau = 0.0

periodic = True


S, Vision_k = path_component_with_occlusion_rays_numba(y0, x0,
                                        dirs_idx,
                                        Pathes,
                                        N_y, N_x,
                                        tau,
                                        dy_rays, dx_rays, w_rays,
                                        periodic)

plt.imshow(Vision_k)
plt.scatter(x0, y0)
plt.show()







