import numpy as np
import os, imageio.v2 as iio
import glob
from pathlib import Path
from probability_model_numba import get_probs_numba_wrapper
from probability_model_numba import dy_rays, dx_rays, w_rays
from probability_model_numba import center_dy_list, center_dx_list
from probability_model_numba import sample_discrete
from core import calculate_distance
from core import get_directions_idx
from core import grow_grass
from core import choose_start_and_destination
from core import los_blocked
from config import e_push
from config import OBST
from plotting import plot_state
from plotting import plot_final_state


# Model parameters: w_1, w_2, c, d
# Pathes parameters: N_x, N_y, V_min, V_max, delta_V, growth_rate


def simulate_trajectory(
        Pathes, 
        PATHES_PARAMS,
        MODEL_PARAMS,
        start_point,
        destination,
        t_max ,
        periodic_boundaries,
        walker_index,
        draw_each_step = False,
        draw_final = False,
        delete_frames = False,
        figdir = "frames"):
    """
    Returns: (trajectory, Pathes_updated)
    
    """
    N_x, N_y = PATHES_PARAMS["N_x"], PATHES_PARAMS["N_y"]
    V_max    = PATHES_PARAMS["V_max"]
    V_min = PATHES_PARAMS["V_min"]
    delta_V  = PATHES_PARAMS["delta_V"]
    d = MODEL_PARAMS["d"]
    eta_obs = MODEL_PARAMS["eta_obs"]
    rho_free = MODEL_PARAMS["rho_free"]
    
    if (draw_each_step or draw_final) and not Path(figdir).exists():
        os.makedirs(figdir)
    frame = 0
    
    # Initialization:
    current_position  = start_point.copy()
    
    current_direction  = 0
    t        = 0
    trajectory     = [current_position.copy()]

    dirs_all = np.arange(8, dtype=np.int32)
    P0 = get_probs_numba_wrapper(dirs_all,
                                 current_position[0], current_position[1],
                                 destination[0], destination[1],
                                 Pathes,
                                 PATHES_PARAMS, MODEL_PARAMS,
                                 periodic_boundaries,
                                 dy_rays, dx_rays, w_rays,
                                 # new:
                                 OBST = OBST, d = d, eta_obs = eta_obs, rho_free = rho_free,
                                 center_dy_list = center_dy_list, center_dx_list = center_dx_list)
    
    current_direction = sample_discrete(P0)
    
    if draw_each_step:
        
        plot_state(Pathes, PATHES_PARAMS, start_point, destination, frame,
                                 f"{figdir}/frame_{frame:05d}.png", walker_index)
        frame += 1

    
    while (calculate_distance(current_position, destination, PATHES_PARAMS, periodic_boundaries) > 1) and (t < t_max):
        
        y, x = current_position
        
        if periodic_boundaries:
            
           x  = x % N_x
        
        if Pathes[y,x] < V_max and Pathes[y, x] >= V_min:
            
            Pathes[y,x] += delta_V * (1 - Pathes[y,x]/V_max)
            Pathes[Pathes[y,x] > V_max] = V_max

        # Available directions:
        
        dirs_idx = get_directions_idx(current_position, current_direction, Pathes, PATHES_PARAMS, periodic_boundaries)
        
        # 3. Probability calculations:
        
        
        at_finish = calculate_distance(current_position, destination, PATHES_PARAMS, periodic_boundaries) <= d
        
        MODEL_PARAMS_step = MODEL_PARAMS.copy()
        
        los_is_blocked = True
        
        if MODEL_PARAMS_step["zero_w2_at_finish"]:
            
           if at_finish:
               
                los_is_blocked = los_blocked(current_position, destination, Pathes)  
                
                #if not los_is_blocked:
                    
                MODEL_PARAMS_step["w_2"] = 0.0
        
        # Here we calculate if there is some building before our destination in visible area. True - if there is some building => blocked
        
        eta_obs   = MODEL_PARAMS_step.get("eta_obs", 0.0)
        rho_free  = MODEL_PARAMS_step.get("rho_free", 0.0)

        if at_finish and not los_is_blocked:
            
                MODEL_PARAMS_step["eta_obs"]  = 0.0
                MODEL_PARAMS_step["rho_free"] = 0.0
            
        P_dirs = get_probs_numba_wrapper(dirs_idx,
                                         current_position[0], current_position[1],
                                         destination[0], destination[1],
                                         Pathes,
                                         PATHES_PARAMS, MODEL_PARAMS_step,
                                         periodic_boundaries,
                                         dy_rays, dx_rays, w_rays,
                                         # new:
                                         OBST = OBST, d = d, eta_obs = eta_obs, rho_free = rho_free,
                                         center_dy_list = center_dy_list, center_dx_list = center_dx_list
                                   )

        # 4. Choice of the new direction:
        
        if at_finish: 
            
            # If the pedestrian is close to the finish, he takes direction with a highest probability:
            
            best_mask = P_dirs == P_dirs.max()
            current_direction = int(np.random.choice(dirs_idx[best_mask]))
            
            
        else:    
            
            current_direction = np.random.choice(dirs_idx, p = P_dirs)

        # 5. Step:
        current_position = current_position + e_push[current_direction]
        
        
        
        if periodic_boundaries:
            current_position[1] %= N_x

        # 6. Grass is growing
        Pathes = grow_grass(Pathes, PATHES_PARAMS)
        
        if draw_each_step:
        
            plot_state(Pathes, PATHES_PARAMS,
                                 start_point, destination, frame,
                                 f"{figdir}/frame_{frame:05d}.png", walker_index)
            frame += 1

        
        trajectory.append(current_position.copy())
        t += 1
    
    if draw_final:
        
        plot_state(Pathes, PATHES_PARAMS,
                                 start_point, destination, step = frame,
                                 fname = f"{figdir}/frame_{walker_index:05d}.png", walker_index = walker_index)
        
        
           
    
    # GIF in case for draw_each_step == True
    
    if draw_each_step:
        files = sorted(Path(figdir).glob("frame_*.png"))
        imgs = [iio.imread(f) for f in files]
        iio.mimsave(f"{figdir}/trajectory.gif", imgs, fps=4)
        
        
    if delete_frames: 
        
        removing_files = glob.glob(f"{figdir}/*.png")
        
        for i in removing_files:
            os.remove(i)    
             
    
    return trajectory, Pathes


    

def simulate_many_walkers(N_walkers,
                          Pathes_init, 
                          PATHES_PARAMS,
                          MODEL_PARAMS,
                          t_max,
                          start_and_destination_points,
                          periodic_boundaries,
                          simulation_number,
                          delete_frames = False,
                          give_only_final_result = True,
                          print_progress = False):
    
    Pathes = Pathes_init.copy()
    trajectories, destinations = [], []
    draw_each_step = False
    draw_final = True
    
    if N_walkers == 1:
        
        draw_each_step = True
        draw_final = False
        delete_final_frame = False
        
    else:
        
        delete_final_frame = False 
        
        if give_only_final_result:
            
            draw_final = False   
        
    for i in range(N_walkers):
        
        if i%100 == 0 and print_progress:
            
            print("Walker: ", i)
            
        start, destination = choose_start_and_destination(start_and_destination_points)
        trajectory, Pathes = simulate_trajectory(Pathes = Pathes, 
                                                PATHES_PARAMS = PATHES_PARAMS,
                                                MODEL_PARAMS = MODEL_PARAMS, 
                                                start_point = start,
                                                destination = destination,
                                                t_max = t_max ,
                                                periodic_boundaries = periodic_boundaries,
                                                draw_each_step = draw_each_step,
                                                draw_final = draw_final,
                                                delete_frames = delete_final_frame,
                                                figdir = "frames",
                                                walker_index = i)
        trajectories.append(trajectory)
        destinations.append(destination)
    
    
    
    if N_walkers > 1:
        
        if give_only_final_result:
            
            plot_final_state(Pathes = Pathes, 
                             PATHES_PARAMS = PATHES_PARAMS, 
                             MODEL_PARAMS = MODEL_PARAMS,
                             N_walkers = N_walkers,
                             simulation_number = simulation_number,
                             )
            
    else: # making gif: 
            
        figdir = "frames"
        files = sorted(Path(figdir).glob("frame_*.png"))
        imgs = [iio.imread(f) for f in files]
        iio.mimsave(f"{figdir}/trajectory.gif", imgs, fps=4)
        
        # deleting files:
        if delete_frames: 
            
            
            
            removing_files = glob.glob(f"{figdir}/frame_*.png")
            
            
            for i in removing_files:
                os.remove(i) 
        
    
    return Pathes, trajectories, destinations    
        
    