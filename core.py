import numpy as np
from config import e_push, MODEL_PARAMS





# Function which chooses start and destination point by random choice for each pedestrian 

def choose_start_and_destination(start_and_destination_points):
    
    indices = np.random.choice(start_and_destination_points.shape[0], 2, replace=False)
    start_point = start_and_destination_points[indices[0]]
    destination_point = start_and_destination_points[indices[1]]
    
    return start_point, destination_point


# Function which calculates distance between two given points p1 and p2, 
# considering the possible inclusion of periodic boundary conditions in x (if periodic_boundaries == True)

def calculate_distance(p1, p2, PATHES_PARAMS, periodic_boundaries):
    
    
    y1, x1 = p1
    y2, x2 = p2
    dx = 0
    N_x = PATHES_PARAMS["N_x"]
    
    if periodic_boundaries:
        
       dx = min(abs(x2 - x1), N_x - abs(x2 - x1))
       
    else: 
        
       dx = abs(x2-x1)   
            
    dy = y2- y1

    distance = np.sqrt(dx**2 + dy**2)
    
    return distance


# --- NEW VERSION of directions choice: ---

# Returns directions idcies which are currently available

def get_directions_idx(current_position, previous_direction_idx, Pathes, PATHES_PARAMS, periodic_boundaries):
    """
    1) We take 3 directions: prev-1, prev, prev+1
    2) Filtration with check_if_directions_are_available → if there are some we return those directions
    3) Othewise we try 5 directions: prev±2 … prev±0
    4) Again filter with check_if_directions_are_available → if there are some we return those directions
    5) Otherwise: try_alternative_directions (the rest)
    """
    # 1) Three candidates:
    
    offsets = [-1, 0, 1]
    candidates = [ (previous_direction_idx + o) % 8 for o in offsets ]
    valid, unavailable = check_if_directions_are_available(candidates,
                                                           current_position,
                                                           Pathes,
                                                           PATHES_PARAMS,
                                                           periodic_boundaries)
    
    if valid.size:
        return valid

    # 3) Now we take 5 directions:
    offsets = [-2, -1, 0, 1, 2]
    candidates = [ (previous_direction_idx + o) % 8 for o in offsets ]
    valid, unavailable = check_if_directions_are_available(candidates,
                                                           current_position,
                                                           Pathes,
                                                           PATHES_PARAMS,
                                                           periodic_boundaries)
    if valid.size:
        return valid

    # 5) The rest:
    return try_alternative_directions(unavailable,
                                      current_position,
                                      Pathes,
                                      PATHES_PARAMS,
                                      periodic_boundaries)





def check_if_directions_are_available(directions, current_position, Pathes, PATHES_PARAMS, periodic_boundaries):
    
    """From directions chooses valid directions, the rest of directions returns as unavailable."""
    
    valid = []
    unavailable = []
        
    for direction in directions:
        if is_valid_direction(direction, current_position, Pathes, PATHES_PARAMS, periodic_boundaries):
            valid.append(direction)
        else:
            unavailable.append(direction)
    return np.array(valid, dtype=int), np.array(unavailable, dtype=int)


def is_valid_direction(direction, current_position, Pathes,
                       PATHES_PARAMS, periodic_boundaries):

    N_x, N_y = PATHES_PARAMS["N_x"], PATHES_PARAMS["N_y"]

    
    step = e_push[direction]
    y = int(current_position[0] + step[0])
    x = int(current_position[1] + step[1])

    
    if periodic_boundaries:
        
        x %= N_x
    
    if (y < 0) or (y >= N_y) or (x < 0) or (x >= N_x) or Pathes[y,x] < -0.5:
        
        return False
    
    else:
        
        return True



def try_alternative_directions(unavailable_directions, current_position, Pathes, PATHES_PARAMS, periodic_boundaries):
    
    """ If main directions are unavailable, we try the rest. """
    
    all_dirs = np.arange(8)
    
    # Exculding:
    
    candidates = [d for d in all_dirs if d not in unavailable_directions]
    valid, _ = check_if_directions_are_available(candidates, current_position, Pathes, PATHES_PARAMS, periodic_boundaries)
    return valid




# ---VISION---

def triangle_vision_in_direction(
    Pathes,
    PATHES_PARAMS,
    MODEL_PARAMS,
    direction_idx,
    current_position,
    periodic_boundaries
):
    
    N_x, N_y = PATHES_PARAMS["N_x"], PATHES_PARAMS["N_y"]
    d = MODEL_PARAMS["d"]
    
    
    
    alpha = direction_idx * np.pi / 4
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    y0, x0 = current_position

    
    Y, X = np.ogrid[0:N_y, 0:N_x]
    dy = Y - y0             # (N_y×1)
    dx = X - x0             # (1×N_x)

    # 3) If periodic:
    if periodic_boundaries:
        dx = (dx + N_x//2) % N_x - N_x//2

    # 4) Rotation
    forward = -dx * sin_a + dy * cos_a
    lateral =  dx * cos_a + dy * sin_a

    # 5) Triangle conditions
    mask = (forward >= 0) & (forward <= d) & (np.abs(lateral) <= forward/2)
    

    # 6) Walls: in Pathes == -1
    mask[Pathes == -1] = False

    # 7) Get coordinates where mask == True:
    ys, xs = np.nonzero(mask)
    
    
    return np.column_stack((ys, xs))


def grow_grass(Pathes, PATHES_PARAMS):
    """
    'Linear'  ➔ Pathes -= growth_rate
    'expo'      ➔ Pathes *= (1 - growth_rate)
    """
    V_min = PATHES_PARAMS["V_min"]
    V_max = PATHES_PARAMS["V_max"]
    decay_type = PATHES_PARAMS["decay_type"]
    growth_rate = PATHES_PARAMS["growth_rate"]
    
    # 1) ,ask (not building and not road):
    mask = (Pathes != -1) & (Pathes != V_max) & (Pathes > V_min)

    # 2) decay:
    if decay_type == "linear":
        Pathes[mask] -= growth_rate
    else:                     
        Pathes[mask] *= (1.0 - growth_rate)

    
    Pathes[mask] = np.maximum(Pathes[mask], V_min)

    
    eps = 1e-6
    Pathes[(Pathes < eps) & (Pathes > 0)] = 0.0

    return Pathes



# New function:

def round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)


class VisionKernel:
    __slots__ = ("dy", "dx", "w")
    def __init__(self, dy, dx, w):
        self.dy = dy.astype(np.int32)
        self.dx = dx.astype(np.int32)
        self.w  = w.astype(np.float32)

def precompute_kernels(d, c):
    
    """
    Returns list of 8 VisionKernel.
    """
    
    kernels = []


    for dir_idx in range(8):
        
        alpha = dir_idx * np.pi / 4
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)

        # local coordinates:
        
        forwards  = np.arange(0, d + 1)
        laterals  = np.arange(-d, d + 1)
        F, L      = np.meshgrid(forwards, laterals, indexing="ij")    

        mask = np.abs(L) <= F / 2
        Fm   = F[mask].astype(float)
        Lm   = L[mask].astype(float)

        # distances in local coordinate system:
        
        dist = np.hypot(Fm, Lm)

        # rotation:
        
        dyf =  Fm * cos_a + Lm * sin_a
        dxf = -Fm * sin_a + Lm * cos_a
        dy  = round_away_from_zero(dyf)
        dx  = round_away_from_zero(dxf)

        w = np.exp(-c * dist)

        kernels.append(VisionKernel(dy, dx, w))

    return kernels


# Kernels precalculation: 

KERNELS = precompute_kernels(MODEL_PARAMS["d"], MODEL_PARAMS["c"])