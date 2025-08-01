#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from PIL import Image
import glob
import os
from config import N_x, N_y, start_and_destination_points, c,d, V_min, V_max, N_walkers, t_max, mapa, delta_V, growth_rate, alpha, beta, e, e_push



start_and_destination_points = np.array(start_and_destination_points)



def choose_start_and_destination(start_and_destination_points):
    # Выбираем случайно два различных индекса из массива
    indices = np.random.choice(start_and_destination_points.shape[0], 2, replace=False)
    start_point = start_and_destination_points[indices[0]]
    destination_point = start_and_destination_points[indices[1]]
    
    return start_point, destination_point


def periodic_distance(p1, p2,N_x):
    
    y1, x1 = p1
    y2, x2 = p2
    
    dx = min(abs(x2 - x1), N_x - abs(x2 - x1))
    dy = y2- y1

    distance = np.sqrt(dx**2 + dy**2)
    
    return distance


def check_if_directions_are_available(directions, current_position, Pathes):
    unavailable_directions = []
    for direction in directions[:]:
            new_position = current_position + e_push[direction]
            new_position[1] %= N_x 
            if Pathes[new_position[0],new_position[1]] == -1:
                directions = np.delete(directions, np.where(directions == direction))
                unavailable_directions.append(direction)
    return np.array(directions), np.array(unavailable_directions)


def try_alternative_directions(unavaiible_directions,current_position, Pathes):
    all_directions = np.arange(8)
    alternative_directions = np.delete(all_directions, unavaiible_directions)
    alternative_directions, unavaiible_directions = check_if_directions_are_available(alternative_directions, current_position, Pathes)
    
    return np.array(alternative_directions)    



    

def get_directions_idx(current_position,previous_direction_idx, Pathes):
    directions = np.zeros(3)
    

    if current_position[0] == 1:
        directions = np.array([6,7,0,1,2])
    if current_position[0] == N_y-2:
        directions = np.array([2,3,4,5,6])     
    else:     
        
        directions = np.array([(previous_direction_idx - 1)%8, previous_direction_idx, (previous_direction_idx + 1)%8])
            
    directions, unavailable_directions = check_if_directions_are_available(directions, current_position, Pathes)
    if directions.shape[0] == 0:
        directions = try_alternative_directions(unavailable_directions, current_position, Pathes)
                  
    return directions

    
    
def get_probabilities(nearby_directions_idx, direction_idx, current_position, destination, Pathes, w_1, w_2, c, d, N_x):  
    nearby_directions = e[nearby_directions_idx]
    
    # Основной вектор
    delta_x_main = destination[1] - current_position[1]
    if abs(delta_x_main) > N_x / 2:
        delta_x_main = delta_x_main - np.sign(delta_x_main) * N_x
    
    delta_y = destination[0] - current_position[0]
    delta_main = np.array([delta_y, delta_x_main])
    
    
    delta_x_left = (destination[1] + N_x) - current_position[1]
    delta_left = np.array([delta_y, delta_x_left])
    
   
    delta_x_right = (destination[1] - N_x) - current_position[1]
    delta_right = np.array([delta_y, delta_x_right])
    
    #odległości
    periodic_dist_main = np.linalg.norm(delta_main)
    periodic_dist_left = np.linalg.norm(delta_left)
    periodic_dist_right = np.linalg.norm(delta_right)
    
    #iloczyny skalarne
    probabilities_distance_main = np.tensordot(nearby_directions, delta_main, axes=(1, 0)) / periodic_dist_main
    probabilities_distance_left = np.tensordot(nearby_directions, delta_left, axes=(1, 0)) / periodic_dist_left
    probabilities_distance_right = np.tensordot(nearby_directions, delta_right, axes=(1, 0)) / periodic_dist_right
    
    
    min_distance = min(periodic_dist_main, periodic_dist_left, periodic_dist_right)
    probabilities_distance = np.zeros_like(probabilities_distance_main)
    if periodic_dist_main == min_distance:
        probabilities_distance += probabilities_distance_main
    if periodic_dist_left == min_distance:
        probabilities_distance += probabilities_distance_left
    if periodic_dist_right == min_distance:
        probabilities_distance += probabilities_distance_right
        
    
    
    probabilities_distance *= probabilities_distance > 0
    
    
    
    if np.sum(probabilities_distance) == 0:
        probabilities_distance += 1  # Добавляем 1, чтобы избежать деления на 0
    probabilities_distance /= np.sum(probabilities_distance)
    probabilities_distance = np.array(np.ma.fix_invalid(probabilities_distance, fill_value=1/np.shape(probabilities_distance)[0]))
    #print("prob distance: ", probabilities_distance)
    
    probabilities_pathes = []
    for direction_idx in nearby_directions_idx:
        coordinates_of_triangle = triangle_vision_in_direction(Pathes, direction_idx,current_position,N_y,N_x, d)
        distances_in_triangle = np.array([periodic_distance(coordinates_of_triangle[i], current_position.T, N_x) for i in range(coordinates_of_triangle.shape[0])])
        probabilities_pathes_in_direction = 0
        for i in range(distances_in_triangle.shape[0]):
            probabilities_pathes_in_direction += np.exp(-c * distances_in_triangle[i]) * Pathes[coordinates_of_triangle[i][0], coordinates_of_triangle[i][1]]
        probabilities_pathes.append(probabilities_pathes_in_direction)
    
    #print(probabilities_pathes)    
    probabilities_pathes = np.array(probabilities_pathes) / np.sum(probabilities_pathes)
    probabilities_pathes = np.array(np.ma.fix_invalid(probabilities_pathes, fill_value=1/np.shape(probabilities_pathes)[0]))
    #print("prob pathes: ", probabilities_pathes)
    
    probabilities = w_1 *probabilities_distance + w_2 * probabilities_pathes 
    probabilities = probabilities / np.sum(probabilities)
    probabilities = np.array(np.ma.fix_invalid(probabilities, fill_value=1/np.shape(probabilities)[0]))
    #print("prob: ", probabilities)
    
    return probabilities

def triangle_vision_in_direction(Pathes, direction_idx,current_position,N_y,N_x, d):   
    direction_idx = direction_idx + int(bool(direction_idx < 4 ) )* 4 - int(bool(direction_idx >= 4 ) )* 4      #przeskalowanie direction_idx
    alpha = direction_idx*np.pi/4   
    y0 = current_position[0]
    x0 = current_position[1]
    tab = np.fromfunction(lambda x, y: \
        np.abs(d - (x - x0) * np.sin(alpha) + (y - y0) * np.cos(alpha) ) + \
        2 * np.abs((x - x0) * np.cos(alpha) + (y - y0) * np.sin(alpha)) < d, (N_x, N_y)) * \
        np.fromfunction(lambda x, y: \
        -(x - x0) * np.sin(alpha) + (y - y0) * np.cos(alpha) > -d, (N_x, N_y))
    tab2 = np.fromfunction(lambda x, y: \
    np.abs(d - (x - x0) % N_x * np.sin(alpha) + (y - y0) * np.cos(alpha) ) + \
    2 * np.abs((x - x0) % N_x * np.cos(alpha) + (y - y0) * np.sin(alpha)) < d, (N_x, N_y)) * \
    np.fromfunction(lambda x, y: \
    -((x - x0) % N_x) * np.sin(alpha) + (y - y0) * np.cos(alpha) > -d, (N_x, N_y))
    tab3 = np.fromfunction(lambda x, y: \
    np.abs(d - (x - x0 - N_x) * np.sin(alpha) + (y - y0) * np.cos(alpha) ) + \
    2 * np.abs((x - x0 - N_x) * np.cos(alpha) + (y - y0) * np.sin(alpha)) < d, (N_x, N_y)) * \
    np.fromfunction(lambda x, y: \
    -(x - x0 - N_x) * np.sin(alpha) + (y - y0) * np.cos(alpha) > -d, (N_x, N_y))
    triangle_vision =  1 * (tab + tab2 + tab3).T
    triangle_vision[Pathes == -1] = 0
    coordinates_of_triangle = np.where(triangle_vision == 1)
    coordinates_of_triangle = list(zip(coordinates_of_triangle[0], coordinates_of_triangle[1]))
    return np.array(coordinates_of_triangle)

def grow_grass(Pathes, growth_rate, V_min, V_max):
    
    grass_indices = (Pathes != -1) & (Pathes != V_max) & (Pathes > V_min)
    Pathes[grass_indices] -= growth_rate
    Pathes[np.round(Pathes,2) == 0] = 0
 
    return Pathes

#Criterion:

def calculate_criterion_on_trajectory(trajectory, destination, Pathes, V_max, V_min, alpha, beta):
    
 
    #Obliczamy odchylenie od najkrótszej ścieżki długość trajektorii:
    
    factor_alpha = 0
    #L_max = np.sqrt(2)*t_max
    
    for i in range(len(trajectory) - 1):
        
        #wektor jednostkowy optymalny:
        
        e_optimal = (destination - trajectory[i])
        e_optimal = e_optimal.astype(float)
        e_optimal /= np.linalg.norm(e_optimal)
        
        #wektor przemieszczenia:
        
        e_i = (trajectory[i+1] - trajectory[i])
        e_i = e_i.astype(float)
        
        #Uwzględniamy warunki periodyczne:
        e_i[1] = (e_i[1] + N_x / 2) % N_x - N_x / 2  
        
        #Normujemy:
        e_i /= periodic_distance(trajectory[i+1], trajectory[i], N_x)
        
        
        factor_alpha += (1 - np.dot(e_i, e_optimal))**2/4 
        if (1 - np.dot(e_i, e_optimal))**2/4 > 1:
            print("warning alfa")
        
    
    
    #print(L_real)    
    #Średni potencjał dla dalnej trajektorii: unormowany
    
    V_diff = 0
    for position in trajectory:
        V_diff += (V_max - Pathes[position[0], position[1]] )**2
        if((V_max - Pathes[position[0], position[1]] )**2/(V_max - V_min)**2 > 1 ):
            print("waning beta")
        
    
    #Kryterium:
    
    
    factor_alpha /= len(trajectory)
    factor_beta = V_diff/(len(trajectory)*(V_max - V_min)**2)
    
    
    criterion = alpha * factor_alpha + beta * factor_beta
   
    
    return criterion, factor_alpha, factor_beta 



def simulate_trajectory(w_1, w_2, c, d, start_point, destination, i, V_max, N_x, t_max, delta_V):
    
    
    global Pathes
    global positions
     
    current_position = start_point.copy()
    current_direction_idx = 0
    t = 0
    
    trajectory = []
    
    probabilities_initial = get_probabilities(np.array([0,1,2,3,4,5,6,7]), current_direction_idx, current_position, destination, Pathes, w_1, w_2,c, d, N_x)                                               
    current_direction_idx = np.random.choice( np.array([0,1,2,3,4,5,6,7]), p = probabilities_initial) 
    trajectory.append(current_position.copy())
    
    while periodic_distance(current_position, destination, N_x) != 0 and t != t_max:
        
        
        t = t + 1    
        #Wydeptanie: 
        positions[current_position[0], current_position[1]] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(Pathes, cmap='hot')
        fig.colorbar(im1, ax=ax1, label="Potential")
        ax1.scatter(start_and_destination_points[:,1],start_and_destination_points[:,0], c = 'blue')
        ax1.scatter(start_point[1], start_point[0], c = "green")
        ax1.scatter(destination_point[1], destination_point[0], c = "red")


        positions_log = np.log1p(positions)
        im2 = ax2.imshow(positions_log, cmap='Greens')
        ax2.set_title('Positions, N_walkers = ' + str(N_walkers))
        fig.colorbar(im2, ax=ax2, label="Visits")
        ax2.scatter(start_and_destination_points[:,1],start_and_destination_points[:,0], c = 'blue')
        ax2.scatter(start_point[1], start_point[0], c = "green")
        ax2.scatter(destination_point[1], destination_point[0], c = "red")
        
        plt.savefig("Pathes_after_" + str(t).zfill(5) +"_steps" + ".png")
        plt.close()
        
        if Pathes[current_position[0], current_position[1]%N_x] < V_max: 
            Pathes[current_position[0], current_position[1]%N_x] += delta_V
            
            
        #Algorytm:
        if periodic_distance(current_position, destination, N_x) <= d: 
            nearby_directions_idx = get_directions_idx(current_position,current_direction_idx, Pathes)  
            probabilities = get_probabilities(nearby_directions_idx, current_direction_idx, current_position, destination, Pathes, w_1, w_2, c, d, N_x)
            current_direction_idx = nearby_directions_idx[np.where(probabilities == np.max(probabilities))][0]
            current_position += e_push[current_direction_idx] 
            current_position[1] %= N_x
            
        else:                                                                                                            
            nearby_directions_idx = get_directions_idx(current_position,current_direction_idx, Pathes)                                                             #Znajdujemy indeksy sąsiędnich kierunków
            probabilities = get_probabilities(nearby_directions_idx, current_direction_idx, current_position, destination, Pathes, w_1, w_2, c, d, N_x)    #Obliczamy prawdopodobieństwa
            current_direction_idx = np.random.choice( nearby_directions_idx, p = probabilities)                                                            #Wybieramy nowy kierunek ruchu
            current_position += e_push[current_direction_idx]                                                                                              #Przesuwamy się o wektor kierunku
            current_position[1] %= N_x
        
        Pathes = grow_grass(Pathes, growth_rate, V_min = V_min, V_max = V_max)                                                                              #Trawa rośnie  
        #Trajektoria:
        trajectory.append(current_position.copy())
        

    
                                                             
    
    return trajectory    




w_1 = 1
w_2 = 1


#Oznaczamy na mapie pozycję początkową oraz punkt docelowy:

print("w_1: ", w_1)
print("w_2 ", w_2)
print("d: ", d)


Pathes = np.load(mapa)

       
positions = np.zeros((N_y, N_x))


#%%



start_point, destination_point = choose_start_and_destination(start_and_destination_points)

trajectory = simulate_trajectory(w_1, w_2, c, d, start_point, destination_point, 0, V_max, N_x, t_max, delta_V)



#Animacja:
    
frames = []
imgs = glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

    
frames[0].save('png_to_gif.gif', format='GIF',append_images=frames[1:],save_all=True,duration=100, loop=0)    


#Usuwanie plików png:

removing_files = glob.glob('*.png')
for i in removing_files:
    os.remove(i)






# %%
