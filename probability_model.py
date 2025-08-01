import numpy as np

from config import e
from core import triangle_vision_in_direction, KERNELS






def get_probabilities(nearby_directions_idx, 
                      current_position, 
                      destination, 
                      Pathes, 
                      PATHES_PARAMS, 
                      MODEL_PARAMS, 
                      periodic_boundaries
                      ):
    
    """
    Returns normalized vector of the probabilities P_i (len = len(nearby_directions_idx)) for a given directions,
    """
    N_x = PATHES_PARAMS["N_x"]
    w_1 = MODEL_PARAMS["w_1"] 
    w_2 = MODEL_PARAMS["w_2"]
    gamma = MODEL_PARAMS["gamma"]
    

    # D-component:
    
    D = distance_component(
            dirs_idx          = nearby_directions_idx,
            current_position  = current_position,
            destination       = destination,
            N_x               = N_x,
            periodic_boundaries = periodic_boundaries
        )                      # shape (k,)

    # S-component:
    
    S = path_component_fast(
            current_position  = current_position,
            dirs_idx          = nearby_directions_idx,
            Pathes            = Pathes,
            PATHES_PARAMS     = PATHES_PARAMS,
            MODEL_PARAMS      = MODEL_PARAMS,
            periodic_boundaries = periodic_boundaries
        )                      # shape (k,)

    # Linear combination and normalization:
    
    if gamma != 1.0:
        
       S = S ** gamma
    
    # 4) Normalization: Σ S_i = 1 
    
    total_S = S.sum()
    
    if total_S > 0:
       S /= total_S
      
    
    P = w_1 * D + w_2 * S
    
    if P.sum() == 0.0:                       
        P[:] = 1.0 / len(P)
    else:
        P /= P.sum()

    return P




# ---Path component S---



def path_component(current_position, dirs_idx, Pathes, PATHES_PARAMS, MODEL_PARAMS, periodic_boundaries):
    
    """
    Returns array S_i of the same length as dirs_idx.
    Expression: S_i = Σ_{p∈Δ_i} V(p) · exp(-c·d_p)   (Δ_i – for direction i)
    """
    y0, x0 = current_position
    S = np.zeros(len(dirs_idx), dtype=float)
    
    N_x = PATHES_PARAMS["N_x"]
    c = MODEL_PARAMS["c"]
    tau = MODEL_PARAMS["tau"]
    

    for k, d_idx in enumerate(dirs_idx):
        
        # 1) We get coordinates of the triangle in given direction d_idx
        
        coords = triangle_vision_in_direction(
            Pathes, PATHES_PARAMS, MODEL_PARAMS, d_idx, current_position, periodic_boundaries
        )
        if coords.size == 0:         
            continue

        ys, xs = coords[:, 0], coords[:, 1]

        #  2) Distances calculation (d_p) in the triangle:
        
        dy = ys - y0
        dx = xs - x0
        
        if periodic_boundaries:
            dx = (dx + N_x // 2) % N_x - N_x // 2   

        dist = np.hypot(dy, dx)                     # √(dx²+dy²) for all dots

        # 3) Sum: V(p)·exp(-c·d_p)
        
        vals = Pathes[ys, xs]                       # V(p)
        S[k] = np.dot(vals, np.exp(-c * dist))      # fast vector calculation
    
    
    local_max = S.max()
    
    if local_max > 0:
        
       S[S < tau*local_max] = 0.0
  
    
    
    return S





def distance_component(dirs_idx, current_position, destination, N_x, periodic_boundaries):
  

    
    dy = destination[0] - current_position[0]
    dx = destination[1] - current_position[1]
    if periodic_boundaries:
        dx = (dx + N_x // 2) % N_x - N_x // 2    
    
    delta = np.array([dy, dx], dtype=float)
    norm  = np.linalg.norm(delta)

    if norm < 1e-9:
        return np.ones(len(dirs_idx), dtype=float) / len(dirs_idx)

    # Projections calculations:
    
    proj = (e[dirs_idx] @ delta) / norm           
    proj[proj < 0] = 0 
    
    if not proj.any():
        
       proj[:] = 1.0 / len(proj)              


    return proj 


### NEW VERSION OF PATH COMPONENT: 

def path_component_fast(current_position, dirs_idx, Pathes,
                   PATHES_PARAMS, MODEL_PARAMS, periodic_boundaries):
    
    """
    Returns not normalized S (len = len(dirs_idx)).
    Works only with compact (vision) KERNELS (см. выше).
    
    """
    y0, x0 = current_position
    N_y, N_x = PATHES_PARAMS["N_y"], PATHES_PARAMS["N_x"]
    tau = MODEL_PARAMS["tau"]

    S = np.zeros(len(dirs_idx), dtype=np.float32)

    for k, d_idx in enumerate(dirs_idx):
        
        ker = KERNELS[d_idx]
        ys = y0 + ker.dy
        xs = x0 + ker.dx

        if periodic_boundaries:
            
            ys %= N_y
            xs %= N_x
            S[k] = np.sum(Pathes[ys, xs] * ker.w)
            
        else:
            
            mask = (ys >= 0) & (ys < N_y) & (xs >= 0) & (xs < N_x)
            if mask.any():
                S[k] = np.sum(Pathes[ys[mask], xs[mask]] * ker.w[mask])

    # τ‑filtering:
    
    if tau > 0.0:
        m = S.max()
        if m > 0.0:
            S[S < tau * m] = 0.0

    return S
