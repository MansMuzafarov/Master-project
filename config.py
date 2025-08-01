import numpy as np



# Simulation_paramters: 

N_walkers = 1

t_max = 1000

# Periodic boundaries in "x": yes or no

periodic_boundaries = False


MODEL_PARAMS = {
    "w_1"  : 1.0,             # distance's part of probability 
    "w_2"  : 2.0,             # vision's part of probability
    "c"    : 0.01,             # large c - less important "far" pathes
    "d"    : 40,              # vision distance
    "gamma": 2.0,             # the degree of "S" factor in probability model
    "tau"  : 0.3,             # mask parameter for "S" factor calculations
    "zero_w2_at_finish": True
}


PATHES_PARAMS = {
    "N_x"                 : 200,
    "N_y"                 : 200,
    "V_min"               : 0.0,
    "V_max"               : 100,                   # asphalt level
    "delta_V"             : 10.0,
    "growth_rate"         : 1e-5,
    "decay_type"          : "linear",                # linear or expo
    "gaussian_initial_map": True,                   # gaussian initial map or uniform
}

HYPERPARAMETERS = {
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.5,
    "thr": 0.2 * PATHES_PARAMS["V_max"]   # for mask creating (in loss function for example)
}

# Map choice: empty_map, snake_map, campus

map_type = 'campus_pathes_map'

mapa = "maps_creating/maps/" + map_type + ".npy"



# Start and destination points type: "triangle", "square", "campus"

start_and_destination_points_type = "campus"       
       
filename = "maps_creating/start_and_destination_points/" + start_and_destination_points_type + ".npy"

if start_and_destination_points_type == "empty":
    np.save(filename, np.array([]))
    


start_and_destination_points = np.load(filename)


# Wektory jednostkowe: (kierunki)
e = np.array([[1, 0], [1/np.sqrt(2), -1/np.sqrt(2)], [0, -1], [-1/np.sqrt(2), -1/np.sqrt(2)], [-1, 0], [-1/np.sqrt(2), 1/np.sqrt(2)], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)]])

e_push = np.array([[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]])


# Saving to maps_creating folder

np.save('maps_creating/grid_params.npy', np.array([PATHES_PARAMS["N_x"], PATHES_PARAMS["N_y"], PATHES_PARAMS["V_max"], PATHES_PARAMS["gaussian_initial_map"], start_and_destination_points_type, map_type]))













