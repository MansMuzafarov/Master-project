from numba import njit
from numba.typed import List
from config import e
from core import KERNELS_RAYS
from core import center_dy_list, center_dx_list
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
    

    for i in range(k):
        
        d_idx = dirs_idx[i]
        rays_dy = dy_rays[d_idx]
        rays_dx = dx_rays[d_idx]
        rays_w  = w_rays [d_idx]

        s_dir = 0.0
        
        for r in range(len(rays_dy)):
            dy = rays_dy[r]; dx = rays_dx[r]; w = rays_w[r]

            
            y_prev = y0
            x_prev = x0

            blocked = False
            for p in range(dy.shape[0]):
                yy = y0 + dy[p]
                xx = x0 + dx[p]

                
                if segment_hits_obstacle(y_prev, x_prev, yy, xx,
                                         Pathes, N_y, N_x, periodic):
                    blocked = True
                    break

                
                if 0 <= yy < N_y:
                    xm = xx % N_x if periodic else xx
                    if 0 <= xm < N_x:
                        s_dir += Pathes[yy, xm] * w[p]
                        
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

    return S


@njit(cache=True)
def get_probabilities_numba(dirs_idx,
                            y0, x0, yd, xd,
                            Pathes,
                            N_y, N_x,
                            w1, w2, gamma, tau,
                            periodic,
                            dy_rays, dx_rays, w_rays,
                            # new:
                            OBST, d, eta_obs, rho_free,
                            center_dy_list, center_dx_list):
    # D
    D = distance_component_numba(dirs_idx, y0, x0, yd, xd, N_x, periodic)

    # S
    S = path_component_with_occlusion_rays_numba(y0, x0, dirs_idx, Pathes,
                                                 N_y, N_x, tau,
                                                 dy_rays, dx_rays, w_rays, periodic)


    if gamma != 1.0:
        for i in range(S.shape[0]):
            S[i] = S[i] ** gamma

    
    ssum = S.sum()
    if ssum > 0:
        S /= ssum

    # New: buildings vision:
    
    B = obstacle_mass_visible_rays_numba(y0, x0, dirs_idx, OBST, N_y, N_x,
                                         dy_rays, dx_rays, w_rays, periodic)
    A_mass = np.exp(-eta_obs * B)

    L = free_clearance_batch_numba(y0, x0, dirs_idx, OBST, N_y, N_x,
                                   center_dy_list, center_dx_list, d, periodic)
    eps = 1e-6
    A_free = ((L + eps) / (d + eps)) ** rho_free
    
    
    P = (w1 * D + w2 * S) * (A_mass * A_free)

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
                            dy_list, dx_list, w_list,
                            # new:
                            OBST, d, eta_obs, rho_free,
                            center_dy_list, center_dx_list):
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
                                   Pathes.astype(np.float32),  
                                   np.int32(N_y), np.int32(N_x),
                                   np.float32(w1), np.float32(w2),
                                   np.float32(gamma), np.float32(tau),
                                   np.uint8(periodic),          
                                   dy_list, dx_list, w_list,
                                   # new:
                                   OBST.astype(np.int32), np.int32(d), np.float32(eta_obs), np.float32(rho_free),
                                   center_dy_list, center_dx_list)



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


# Buildings vision:

@njit(cache=True)
def obstacle_mass_visible_rays_numba(y0, x0,
                                     dirs_idx,
                                     OBST,
                                     N_y, N_x,
                                     dy_rays, dx_rays, w_rays,
                                     periodic):
    k = dirs_idx.shape[0]
    B = np.zeros(k, dtype=np.float32)

    for i in range(k):
        d = dirs_idx[i]
        rays_dy = dy_rays[d]
        rays_dx = dx_rays[d]
        rays_w  = w_rays [d]

        s = 0.0
        for r in range(len(rays_dy)):
            dy = rays_dy[r]
            dx = rays_dx[r]
            w  = rays_w [r]

            for p in range(dy.shape[0]):
                yy = y0 + dy[p]
                xx = x0 + dx[p]

                if 0 <= yy < N_y:
                    xm = xx % N_x if periodic else xx
                    if 0 <= xm < N_x:
                        if OBST[yy, xm] == 1:
                            s += w[p]   # only first layer of the building
                            break       # and we kill ONLY this ray

        B[i] = s

    return B

@njit(cache=True)
def free_clearance_batch_numba(y0, x0, dirs_idx, OBST, N_y, N_x,
                               center_dy_list, center_dx_list, d, periodic):
    k = dirs_idx.shape[0]
    L = np.zeros(k, dtype=np.float32)
    for i in range(k):
        d_idx = dirs_idx[i]
        ray_dy = center_dy_list[d_idx]
        ray_dx = center_dx_list[d_idx]
        free_len = 0
            
        for t in range(ray_dy.shape[0]):
            yy = y0 + ray_dy[t]
            xx = x0 + ray_dx[t]
            if periodic:
                xx %= N_x
            if not (0 <= yy < N_y and 0 <= xx < N_x):
                break
            if OBST[yy, xx] == 1:
                break
            free_len += 1
              
        L[i] = free_len
    return L



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



def pack_rays_for_numba(kerns_py):
    
    dy_lists = List(); dx_lists = List(); w_lists = List()
    
    for K in kerns_py:
        dyrs = List(); dxrs = List(); wrs = List()
        for j in range(len(K.rays_dy)):
            dyrs.append(np.asarray(K.rays_dy[j], dtype=np.int32))
            dxrs.append(np.asarray(K.rays_dx[j], dtype=np.int32))
            wrs .append(np.asarray(K.rays_w [j], dtype=np.float32))
        dy_lists.append(dyrs); dx_lists.append(dxrs); w_lists.append(wrs)
        
    return dy_lists, dx_lists, w_lists

# Kernels precalculation for numba: 

dy_rays, dx_rays, w_rays = pack_rays_for_numba(KERNELS_RAYS)

# Rays precalculation for numba (buildings vision):

center_dy_list = List([np.asarray(v, dtype=np.int32) for v in center_dy_list])
center_dx_list = List([np.asarray(v, dtype=np.int32) for v in center_dx_list])

