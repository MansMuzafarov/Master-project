import numpy as np
import matplotlib.pyplot as plt

from config import PATHES_PARAMS, MODEL_PARAMS, HYPERPARAMETERS
from config import mapa
from config import N_walkers, t_max, periodic_boundaries, start_and_destination_points
from simulate_trajectory import simulate_many_walkers
from Active_Walker_optimization import optimize_model_grid


# Initial map:

initial_map = np.load(mapa)

# Optimization parameters:


target_map  = np.load("target_map/target_map_square.npy")

# Choose here what you would like to do:

optimize = False
make_simulation = True




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
                                                                simulation_number = 1,
                                                                delete_frames = False,
                                                                give_only_final_result = True,
                                                                print_progress = True)



    
   