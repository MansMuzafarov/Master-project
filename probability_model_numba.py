from numba import njit
from numba.typed import List
from config import e
from core import KERNELS
import numpy as np



@njit(cache=True)
def distance_component_numba(dirs_idx,
                             y0, x0, yd, xd,
                             N_x, periodic):
 
    dy = yd - y0
    dx = xd - x0
    if periodic:
        half = N_x // 2
        if dx >  half: dx -= N_x
        if dx < -half: dx += N_x

    norm = np.sqrt(dy*dy + dx*dx)
    k = dirs_idx.shape[0]
    out = np.zeros(k, dtype=np.float32)

    if norm < 1e-12:
        
        out[:] = 1.0 / k
        return out

    for i in range(k):
        d_idx = dirs_idx[i]
        vy = e[d_idx, 0]
        vx = e[d_idx, 1]
        pr = (vy * dy + vx * dx) / norm
        if pr > 0.0:
            out[i] = pr
        else:
            out[i] = 0.0
    return out

def kernels_to_numba_lists(kernels_py):
    """ Упаковываем в numba.typed.List для нативной работы в @njit """
    dy_list = List()
    dx_list = List()
    w_list  = List()
    for K in kernels_py:
        dy_list.append(np.asarray(K.dy, dtype=np.int32))
        dx_list.append(np.asarray(K.dx, dtype=np.int32))
        w_list .append(np.asarray(K.w , dtype=np.float32))
    return dy_list, dx_list, w_list

@njit(cache=True)
def path_component_fast_numba(y0, x0,
                              dirs_idx,
                              Pathes,
                              N_y, N_x,
                              tau,
                              dy_list, dx_list, w_list,
                              periodic):
    
    """
    Not normalized S-component, uses compact kernels.
    """
    k = dirs_idx.shape[0]
    S = np.zeros(k, dtype=np.float32)

    for i in range(k):
        d_idx = dirs_idx[i]
        dy = dy_list[d_idx]
        dx = dx_list[d_idx]
        w  = w_list [d_idx]

        s = 0.0
        for p in range(dy.shape[0]):
            yy = y0 + dy[p]
            xx = x0 + dx[p]

            if periodic:
                yy_mod = yy
                xx_mod = xx % N_x
                if 0 <= yy_mod < N_y:
                    v = Pathes[yy_mod, xx_mod]
                    if v != -1.0:
                        s += v * w[p]
            else:
                if 0 <= yy < N_y and 0 <= xx < N_x:
                    v = Pathes[yy, xx]
                    if v != -1.0:
                        s += v * w[p]

        S[i] = s

    if tau > 0.0:
        m = 0.0
        for i in range(k):
            if S[i] > m:
                m = S[i]
        if m > 0.0:
            thr = tau * m
            for i in range(k):
                if S[i] < thr:
                    S[i] = 0.0

    return S


@njit(cache=True)
def get_probabilities_numba(dirs_idx,
                            y0, x0, yd, xd,
                            Pathes,
                            N_y, N_x,
                            w1, w2, gamma, tau,
                            periodic,
                            dy_list, dx_list, w_list):
    """
    P_i = w1*D + w2*S^gamma and normalization.
    """
    # D
    D = distance_component_numba(dirs_idx, y0, x0, yd, xd, N_x, periodic)

    # S
    S = path_component_fast_numba(y0, x0, dirs_idx, Pathes,
                                  N_y, N_x, tau,
                                  dy_list, dx_list, w_list,
                                  periodic)

    # gamma
    if gamma != 1.0:
        for i in range(S.shape[0]):
            S[i] = S[i] ** gamma

    # нормируем S (опционально — как вы решили)
    ssum = S.sum()
    if ssum > 0:
        S /= ssum

    # итог
    P = w1 * D + w2 * S
    psum = P.sum()

    if psum <= 1e-12:
        P[:] = 1.0 / P.shape[0]
    else:
        P /= psum

    return P

def get_probs_numba_wrapper(dirs_idx,
                            y0, x0, yd, xd,
                            Pathes,
                            PATHES_PARAMS, MODEL_PARAMS,
                            periodic,
                            dy_list, dx_list, w_list):
    N_y = PATHES_PARAMS["N_y"]
    N_x = PATHES_PARAMS["N_x"]

    w1    = MODEL_PARAMS["w_1"]
    w2    = MODEL_PARAMS["w_2"]
    gamma = MODEL_PARAMS["gamma"]
    tau   = MODEL_PARAMS["tau"]

    # numba любит int32
    dirs_idx = np.asarray(dirs_idx, dtype=np.int32)

    return get_probabilities_numba(dirs_idx,
                                   np.int32(y0), np.int32(x0),
                                   np.int32(yd), np.int32(xd),
                                   Pathes.astype(np.float32),  # если надо
                                   np.int32(N_y), np.int32(N_x),
                                   np.float32(w1), np.float32(w2),
                                   np.float32(gamma), np.float32(tau),
                                   np.uint8(periodic),          # бул в int8
                                   dy_list, dx_list, w_list)



@njit(cache=True)
def sample_discrete(P):
    
    """
    Analogy of p = (...)
    """
    r = np.random.random()
    acc = 0.0
    for i in range(P.shape[0]):
        acc += P[i]
        if r <= acc:
            return i
    return P.shape[0] - 1



# Kernels precalculation for numba: 

dy_list, dx_list, w_list = kernels_to_numba_lists(KERNELS)