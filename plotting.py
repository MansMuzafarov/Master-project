import numpy as np
import matplotlib.pyplot as plt



def plot_state(Pathes, PATHES_PARAMS,
                             start_point, destination, step,
                             fname, walker_index):
    
    V_max = PATHES_PARAMS["V_max"]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(Pathes / V_max, cmap="hot", origin="upper", vmin = 0, vmax = 1)
    ax.set_title(f"Step {step}, for walker {walker_index}.")
    fig.colorbar(im, ax=ax, shrink=0.8)
    

    ax.scatter(start_point[1], start_point[0], c='lime', marker='o')
    ax.scatter(destination[1], destination[0], c='red', marker='x')
    

    plt.savefig(fname, dpi=150)
    plt.close(fig)



def plot_final_state(Pathes, MODEL_PARAMS, PATHES_PARAMS, N_walkers, simulation_number):
    
    V_max = PATHES_PARAMS["V_max"]
    
    w_2 = MODEL_PARAMS["w_2"]
    gamma = MODEL_PARAMS["gamma"]
    c = MODEL_PARAMS["c"]
    d = MODEL_PARAMS["d"]
    tau = MODEL_PARAMS["tau"]
    simulation_number_str = str(simulation_number)
    
    # Right: Pathes
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(Pathes / V_max, cmap="hot", origin="upper", vmin = 0, vmax = 1)
    ax.set_title(f"N_walkers = {N_walkers} w₂={w_2}, gamma={gamma}, d = {d}, tau = {tau}, c = {c}.")
    fig.colorbar(im, ax=ax, shrink=0.8)
   
    figdir = "frames"
    fname = f"{figdir}/final_result_" + simulation_number_str + ".png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)