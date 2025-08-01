import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from PIL import Image
import glob
import os
from config import N_x, N_y, y_s, x_s, y_D, x_D, c,d, V_min, V_max, N_walkers, t_max





#Wektory jednostkowe: (kierunki)
e = np.array([[1, 0], [1/np.sqrt(2), -1/np.sqrt(2)], [0, -1], [-1/np.sqrt(2), -1/np.sqrt(2)], [-1, 0], [-1/np.sqrt(2), 1/np.sqrt(2)], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)]])

e_push = np.array([[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]])



def periodic_distance(p1, p2,N_x):
    
    y1, x1 = p1
    y2, x2 = p2
    
    dx = min(abs(x2 - x1), N_x - abs(x2 - x1))
    dy = y2- y1

    distance = np.sqrt(dx**2 + dy**2)
    
    return distance




def get_directions_idx(current_position,direction_idx):
    directions = np.zeros(3)

    if current_position[0] == 1:
        directions = np.array([6,7,0,1,2])
    if current_position[0] == N_y-2:
        directions = np.array([2,3,4,5,6])     
    else:     
        
        directions = np.array([(direction_idx - 1)%8, direction_idx, (direction_idx + 1)%8])
    return directions

def get_neighbors(current_position, direction_idx):
    directions = e_push[get_directions_idx(current_position,direction_idx) ]
    neighbors = current_position + directions
    return neighbors

    
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
        coordinates_of_triangle = triangle_vision_in_direction(direction_idx, current_position, N_y, N_x, d)
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

def triangle_vision_in_direction(direction_idx,current_position,N_y,N_x, d):   
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
    coordinates_of_triangle = np.where(triangle_vision == 1)
    coordinates_of_triangle = list(zip(coordinates_of_triangle[0], coordinates_of_triangle[1]))
    return np.array(coordinates_of_triangle)

def grow_grass(Pathes, growth_rate, V_min, V_max):
    road_potential= V_max
    grass_indices = (Pathes != road_potential)
    Pathes[grass_indices & (Pathes > V_min)] -= growth_rate
    Pathes[Pathes < V_min] = V_min  
    return Pathes

#Criterion:

def calculate_criterion_on_trajectory(start_position, destination, trajectory, Pathes, V_max, V_min):
    
    L_optimal = periodic_distance(np.array(start_position), np.array(destination), N_x)
    e_optional = (destination - start_position)/L_optimal
    
    #Obliczamy długość trajektorii:
    
    factor_alpha = 0
    #L_max = np.sqrt(2)*t_max
    
    for i in range(len(trajectory) - 1):
        
    
        e_i = (trajectory[i+1] - trajectory[i])
        e_i = e_i.astype(float)
        e_i[1] /= N_x
        e_i /= periodic_distance(trajectory[i+1], trajectory[i], N_x)
        factor_alpha += (1 - np.dot(e_i, e_optional))**2/4 
        
        
    
  
    
    V_diff = 0
    for position in trajectory:
        V_diff += (V_max - Pathes[position[0], position[1]] )**2
        
    
    
    #Waga dla długości ścieżki:
    alpha = 0.5     
    
    #Waga dla potencjału ścieżki:
    beta = 0.5
    
    #Kryterium:
    
    
    factor_alpha /= len(trajectory)
    factor_beta = V_diff/(len(trajectory)*(V_max - V_min)**2)
    
    
    criterion = alpha * factor_alpha + beta * factor_beta
   
    
    return criterion, factor_alpha, factor_beta  



def simulate_trajectory(w_1, w_2, c, d, start_point, destination, i, V_max, N_x, t_max):
    
    
    global Pathes
    global positions
     
    current_position = start_point.copy()
    current_direction_idx = 4
    t = 0
    
    trajectory = []
    
    probabilities_initial = get_probabilities(np.array([0,1,2,3,4,5,6,7]), current_direction_idx, current_position, destination, Pathes, w_1, 0,c, d, N_x) 
    current_direction_idx = np.random.choice( np.array([0,1,2,3,4,5,6,7]), p = probabilities_initial) 
    trajectory.append(current_position.copy())
    
    while periodic_distance(current_position, destination, N_x) != 0 and t != t_max:
        
    
        t = t + 1    
        #Wydeptanie: 
        positions[current_position[0], current_position[1]] += 1
        
        if Pathes[current_position[0], current_position[1]%N_x] < V_max: 
            Pathes[current_position[0], current_position[1]%N_x] += 10
            
            
        #Algorytm:
        if periodic_distance(current_position, destination, N_x) <= d: 
            nearby_directions_idx = get_directions_idx(current_position,current_direction_idx)  
            probabilities = get_probabilities(nearby_directions_idx, current_direction_idx, current_position, destination, Pathes, w_1, 0, c, d, N_x)
            current_direction_idx = nearby_directions_idx[np.where(probabilities == np.max(probabilities))][0]
            current_position += e_push[current_direction_idx] 
            current_position[1] %= N_x
            
        else:                                                                                                            
            nearby_directions_idx = get_directions_idx(current_position,current_direction_idx)                                                             #Znajdujemy indeksy sąsiędnich kierunków
            probabilities = get_probabilities(nearby_directions_idx, current_direction_idx, current_position, destination, Pathes, w_1, w_2, c, d, N_x)    #Obliczamy prawdopodobieństwa
            current_direction_idx = np.random.choice( nearby_directions_idx, p = probabilities)                                                            #Wybieramy nowy kierunek ruchu
            current_position += e_push[current_direction_idx]                                                                                              #Przesuwamy się o wektor kierunku
            current_position[1] %= N_x
        
        Pathes = grow_grass(Pathes, growth_rate = 0.001, V_min = 0, V_max = V_max)                                                                              #Trawa rośnie  
        #Trajektoria:
        trajectory.append(current_position.copy())
    #Obrazek
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(Pathes/V_max, cmap="hot")
    # plt.scatter(start_point[1], start_point[0],c ='g' )
    # plt.scatter(destination[1],destination[0], c = 'r')
    # plt.colorbar()
    # plt.title("Numer przechodnia: " + str(i))
    # plt.savefig("step" + str(i).zfill(5) + ".png")
    # plt.close()                                                              
    
    return trajectory    

    


                            
destination = np.array([y_D, x_D])

start_point = np.array([y_s, x_s])

current_position = start_point

start_point = np.array([y_s, x_s])
current_direction_idx = 4


#Ścieżki:

Pathes = np.load('pathes_crossroads.npy')
positions = np.zeros((N_y, N_x))


w_1 = 1
w_2 = 2



print("w_1: ", w_1)
print("w_2 ", w_2)
print("d: ", d)



criteria = []
factors_alpha = []
factors_beta = []


#Główna pętla po przechodniach:
for i in range(N_walkers):
            
    trajectory_i = simulate_trajectory(w_1, w_2, c, d, start_point, destination, i, V_max, N_x, t_max)
    criterion_i, factor_alpha_i, factor_beta_i = calculate_criterion_on_trajectory(start_point, destination, trajectory_i, Pathes, V_max, V_min)
    criteria.append(criterion_i)
    factors_alpha.append(factor_alpha_i)
    factors_beta.append(factor_beta_i)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(Pathes, cmap='hot')
    fig.colorbar(im1, ax=ax1, label="Potential")
    ax1.scatter(start_point[1], start_point[0],c ='g' )   
    ax1.scatter(destination[1],destination[0], c = 'r')
    ax1.set_title("Potential map after: " + str(i) +" walkers") 
    
    positions_log = np.log1p(positions)
    im2 = ax2.imshow(positions_log, cmap='Greens')
    ax2.set_title("Positions map after: " + str(i) +" walkers" )
    fig.colorbar(im2, ax=ax2, label="Visits")
    
    
    plt.savefig("Pathes_after_" + str(i).zfill(5) +"_walkers" ".png")
    plt.close()
    
    
   
# plt.scatter(factors_alpha, factors_beta)
# plt.xlabel("factors alpha")
# plt.ylabel("factors beta")
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.savefig("Factors for w_1 = " + str(w_1) + ", w_2 = " + str(w_2) + ".png")
# plt.close()
        
        
avg_criterion = np.mean(criteria)
print("avg criterion: ", avg_criterion)     




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

