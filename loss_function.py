import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize, binary_opening
from scipy.ndimage import distance_transform_edt
from config import PATHES_PARAMS



def loss_function(Pathes_sim, target_map,
                  V_max,
                  thr,
                  lambda_iou=0.5, lambda_ch=0.3, lambda_mse=0.2, lambda_K=0.0,
                  K_vals=None):
    
    """Returns scalar loss"""
    
    # binary masks:
    
    mask_sim = Pathes_sim >= thr
    mask_tgt = target_map  >= thr
    
    # noise deleting:
    
    mask_sim = binary_opening(mask_sim)

    # IoU:
    
    inter = np.logical_and(mask_sim, mask_tgt).sum()
    union = np.logical_or (mask_sim, mask_tgt).sum()
    iou_loss = 1.0 - inter/union if union else 1.0

    # Chamfer (skeleton-shape):
    
    sk_sim = skeletonize(mask_sim)
    sk_tgt = skeletonize(mask_tgt)
    if sk_sim.any() and sk_tgt.any():
        d_tgt = distance_transform_edt(~sk_tgt)
        d_sim = distance_transform_edt(~sk_sim)
        d1 = d_tgt[sk_sim].mean()
        d2 = d_sim[sk_tgt].mean()
        chamfer = 0.5*(d1+d2) / max(Pathes_sim.shape)
    else:
        chamfer = 1.0

    # MSE:
    
    roi = np.logical_or(mask_sim, mask_tgt)
    if roi.any():
        mse = np.mean(((Pathes_sim - target_map)[roi]/V_max)**2)
    else:
        mse = 1.0

    # K (trajectory loss - optional):
    
    K_loss = np.mean(K_vals) if (lambda_K > 0 and K_vals is not None) else 0.0

    return lambda_iou*iou_loss + lambda_ch*chamfer + lambda_mse*mse + lambda_K*K_loss 




def trajectory_loss_K(trajectory,
                      destination,
                      Pathes,
                      PATHES_PARAMS,
                      HYPERPARAMETERS,
                      periodic_boundaries):
   
    N_x = PATHES_PARAMS["N_x"]
    V_min = PATHES_PARAMS["V_min"]
    V_max = PATHES_PARAMS["V_max"]
    alpha = HYPERPARAMETERS["alpha"]
    beta = HYPERPARAMETERS["beta"]
    L = len(trajectory) - 1
                             
    if L <= 0:
        return 0.0

    # ------- α-component ----------------------------------
    alpha_sum = 0.0
    for i in range(L):
        
        dy = destination[0] - trajectory[i][0]
        dx = destination[1] - trajectory[i][1]
        if periodic_boundaries and N_x is not None:
            dx = (dx + N_x // 2) % N_x - N_x // 2
        e_opt = np.array([dy, dx], dtype=float)
        e_opt /= np.linalg.norm(e_opt)

        
        step = trajectory[i + 1] - trajectory[i]
        if periodic_boundaries and N_x is not None:
            step[1] = (step[1] + N_x // 2) % N_x - N_x // 2
        step = step.astype(float)
        norm = np.linalg.norm(step) or 1e-9
        step /= norm

        alpha_sum += (1.0 - np.dot(step, e_opt)) / 2.0

    factor_alpha = alpha_sum / L                   # ∈ [0,1]

    # ------- β-component --------------------- (now linear)
    beta_sum = 0.0
    for y, x in trajectory:
        beta_sum += (V_max - Pathes[y, x]) 
    factor_beta = beta_sum / (L * (V_max - V_min))   # ∈ [0,1]

    return alpha * factor_alpha + beta * factor_beta





