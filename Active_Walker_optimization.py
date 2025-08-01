import optuna, numpy as np, json
import matplotlib.pyplot as plt
from simulate_trajectory import simulate_many_walkers
from loss_function import loss_function
from config import MODEL_PARAMS, PATHES_PARAMS
from config import N_walkers, t_max




# Grid optimization: 


def make_params(w_2):
    
    """Returns MODEL_PARAMS with w₂, gamma, growth_rate."""
    
    params_model = MODEL_PARAMS.copy()
    params_model["w_2"]   = float(w_2)
    
    
    return params_model


# ---------------------------------------------------------------
# 2)  Grid‑search
# ---------------------------------------------------------------
def optimize_model_grid(initial_map,          # np.ndarray  (N_y×N_x)
                        target_map,           # np.ndarray
                        PATHES_PARAMS,
                        start_and_destination_points,
                        HYPERPARAMETERS,
                        grid_w2,              # iterable
                        keep_best           = 5,
                        t_max               = t_max,
                        periodic_boundaries = False):

    V_max = PATHES_PARAMS["V_max"]
    gamma = MODEL_PARAMS["gamma"]
    d = MODEL_PARAMS["d"]
    tau = MODEL_PARAMS["tau"]
    c = MODEL_PARAMS["c"]
    thr   = HYPERPARAMETERS["thr"]

    candidates = []   # [(loss, param_dict), …]

    sim_idx = 0
    for w2 in grid_w2:
        
            
                    params_model= make_params(w2)
                    sim_idx += 1
                    print(f"\n▶ simulation #{sim_idx} w₂={w2:.2f}  γ={gamma:.2f} d = {d:.2f} tau = {tau:.2f}, c = {c:.2f} ")

                    sim_map, trajectories, _ = simulate_many_walkers(
                            N_walkers                    = N_walkers,
                            Pathes_init                  = initial_map.copy(),
                            PATHES_PARAMS                = PATHES_PARAMS,
                            MODEL_PARAMS                 = params_model,
                            t_max                        = t_max,
                            start_and_destination_points = start_and_destination_points,
                            simulation_number            = sim_idx,
                            periodic_boundaries          = periodic_boundaries,
                            delete_frames                = False,
                            give_only_final_result       = True,
                            print_progress               = False)

                    L = loss_function(sim_map, target_map, V_max, thr,
                                    lambda_iou=0.5, lambda_ch=0.3, lambda_mse=0.2)

                    candidates.append((L, params_model))
                    print(f"   loss = {L:.4f}")

    # sorting:
    candidates.sort(key=lambda tpl: tpl[0])
    top = candidates[:keep_best]

    print("\n=== TOP‑{} configurations ===".format(keep_best))
    for rank, (L, par) in enumerate(top, 1):
        print(f"{rank:2d}) loss={L:.4f}   w₂={par['w_2']:.2f} tau = {par['d']:.2f}  γ={par['gamma']:.2f}  τ={par['tau']:.2f}")

    best_loss, best_params = top[0]
    return best_params, best_loss, top













def sample_random(w_2):
        
    " returns dict(w1,w2,c,d,gamma,tau) из huge range: "
    
    parameters_sample = MODEL_PARAMS.copy()
    parameters_sample["w_2"] = w_2
    
    return parameters_sample




