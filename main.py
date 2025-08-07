import numpy as np
from config import PATHES_PARAMS, MODEL_PARAMS, HYPERPARAMETERS
from config import Pathes_init, target_map
from config import N_walkers, t_max, periodic_boundaries, start_and_destination_points
from simulate_trajectory import simulate_many_walkers
from Active_Walker_optimization import optimize_model_grid
from plotting import plot_final_state
from loss_function import loss_function


# Initial map:

initial_map = Pathes_init.copy()

# Target map:

target_map = target_map.copy()

# Choose here what you would like to do:

optimize = False

make_simulation = True

averaging = False



if optimize:
    
    GRID_W2     = [4.5]
    
    
    best_parameters =   optimize_model_grid(initial_map,          # np.ndarray  (N_y×N_x)
                                            target_map,           # np.ndarray
                                            PATHES_PARAMS,
                                            start_and_destination_points,
                                            HYPERPARAMETERS,
                                            grid_w2 = GRID_W2,               # iterable
                                            keep_best   = 5,
                                            t_max       = 1000,
                                            periodic_boundaries = False)
                                      



if make_simulation:
    
    
    Pathes, trajectories, destinations = simulate_many_walkers(N_walkers = N_walkers,
                                                               Pathes_init = initial_map, 
                                                               PATHES_PARAMS = PATHES_PARAMS,
                                                               MODEL_PARAMS = MODEL_PARAMS,
                                                               t_max = t_max,
                                                               start_and_destination_points = start_and_destination_points,
                                                               periodic_boundaries = periodic_boundaries,
                                                               simulation_number = 0,
                                                               delete_frames = True,
                                                               give_only_final_result = True,
                                                               print_progress = True)


N_averaging = 5

if averaging:
    
    V_max = PATHES_PARAMS["V_max"]
    thr = HYPERPARAMETERS["thr"]
    
    Pathes_final_list = []
    
    for i in range(N_averaging):
        
        Pathes, trajectories, destinations = simulate_many_walkers(N_walkers = N_walkers,
                                                                Pathes_init = initial_map, 
                                                                PATHES_PARAMS = PATHES_PARAMS,
                                                                MODEL_PARAMS = MODEL_PARAMS,
                                                                t_max = t_max,
                                                                start_and_destination_points = start_and_destination_points,
                                                                periodic_boundaries = periodic_boundaries,
                                                                simulation_number = i,
                                                                delete_frames = False,
                                                                give_only_final_result = True,
                                                                print_progress = True)
        
        Pathes_final_list.append(Pathes)
    
    # Averaging:
    
    Pathes_average = np.mean(Pathes_final_list, axis = 0)
    
    # Loss-function: 
    L_average = loss_function(Pathes_sim = Pathes_average, target_map = target_map, V_max = V_max, thr = thr,
                                    lambda_iou=0.5, lambda_ch=0.3, lambda_mse=0.2)
    
    L_sample = loss_function(Pathes_sim = Pathes_final_list[0], target_map = target_map, V_max = V_max, thr = thr,
                                    lambda_iou=0.5, lambda_ch=0.3, lambda_mse=0.2)
    
    print("Loss function for averaged map: ", L_average)
    print("Loss function for the first simulation: ", L_sample)
    
    # Plotting averaged result:
    
    plot_final_state(Pathes = Pathes_average, 
                     PATHES_PARAMS = PATHES_PARAMS, 
                     MODEL_PARAMS = MODEL_PARAMS,
                     N_walkers = N_walkers,
                     simulation_number = "averaged",
                    )
   