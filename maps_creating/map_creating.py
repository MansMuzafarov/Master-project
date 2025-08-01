import pygame
import numpy as np
import os


N_x, N_y, V_max, gaussian_initial_map, start_and_destination_points_type, map_type = np.load('grid_params.npy')

print(start_and_destination_points_type)
N_x   = int(N_x)
N_y   = int(N_y)
V_max = int(V_max)

cell_size = 4  # Размер ячейки в пикселях
screen_width = N_x * cell_size
screen_height = N_y * cell_size
BUILDING_VALUE = -1  # Потенциал зданий

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)  # Цвет для добавляемых вручную точек старта и цели

# Проверяем, существует ли сохранённая карта
pathes_file = 'maps/' + map_type + '.npy'
start_and_destination_file = 'start_and_destination_points/' + start_and_destination_points_type + '.npy'

def show_coordinates_on_screen(screen, coordinates):
    text = font.render(f'Coordinates: {coordinates}', True, (0, 0, 0))
    screen.blit(text, (10, 10))

if os.path.exists(pathes_file):
    
    Pathes = np.load(pathes_file)
    
else:
    
    if gaussian_initial_map:  
        
       Pathes = np.random.normal(loc=10, scale=1, size=(N_y, N_x))  # базовое значение потенциала
       print("Pathes: ", Pathes)
    else:  
         
       Pathes = np.zeros((N_y,N_x))
       

# Проверяем, существует ли сохранённый массив точек старта и цели
if os.path.exists(start_and_destination_file):
    start_and_destination_points = np.load(start_and_destination_file).tolist()
else:
    start_and_destination_points = []


V_min = 0    # Минимальная протоптанность для сброса дорожек

# Инициализация pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw Asphalt Roads and Start/End Points")
font = pygame.font.SysFont(None, 24)

# Инициализация координат для отображения
grid_x, grid_y = 0, 0  # Инициализация переменных

def draw_grid():
    for x in range(0, screen_width, cell_size):
        for y in range(0, screen_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, GRAY, rect, 1)

def draw_roads(Pathes):
    for y in range(N_y):
       for x in range(N_x):
            if Pathes[y, x] == V_max:  # Дорожки
                color = (0, 0, 0)  # Черный для дорожек
            elif Pathes[y, x] == BUILDING_VALUE:  # Здания
                color = (255, 0, 0)  # Красный для зданий
            else:
                color = (255, 255, 255)  # Другой цвет для областей с другим потенциалом
            pygame.draw.rect(screen, color, (x * cell_size, y * cell_size, cell_size, cell_size))

    # Рисуем вручную добавленные точки старта и цели
    for point in start_and_destination_points:
        pygame.draw.circle(screen, BLUE, (point[1] * cell_size + cell_size // 2, point[0] * cell_size + cell_size // 2), cell_size // 2)

# Проверка, находится ли координата в пределах области карты
def is_within_bounds(x, y):
    return 0 <= x < N_x and 0 <= y < N_y

# Функция для удаления ближайшей точки старта или цели
def remove_nearest_point(grid_x, grid_y):
    closest_point = None
    min_dist = float('inf')
    for point in start_and_destination_points:
        dist = (point[0] - grid_y) ** 2 + (point[1] - grid_x) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_point = point

    # Удаляем ближайшую точку, если она достаточно близка
    if closest_point and min_dist < 1:  # Подстраиваем порог на усмотрение
        start_and_destination_points.remove(closest_point)

# Функция для добавления точки старта или цели, если она ещё не существует
def add_unique_point(grid_y, grid_x):
    if [grid_y, grid_x] not in start_and_destination_points:
        start_and_destination_points.append([grid_y, grid_x])

# Главный цикл
running = True
drawing = False
BUILDING_VALUE = -1  # Потенциал зданий

current_mode = 'road'  # Начальный режим рисования - дорожки

# Модифицированный основной цикл
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # Нажимаем 'r' для режима рисования дорожек
                current_mode = 'road'
            elif event.key == pygame.K_e:  # Нажимаем 'e' для режима удаления дорожек
                current_mode = 'erase'
            elif event.key == pygame.K_b:  # Нажимаем 'b' для режима добавления зданий
                current_mode = 'building'
            elif event.key == pygame.K_s:  # Нажимаем 's' для добавления точки старта или цели
                current_mode = 'start_or_end'
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            grid_x = x // cell_size
            grid_y = y // cell_size
            if is_within_bounds(grid_x, grid_y):
                if event.button == 1:  # Левая кнопка мыши
                    drawing = True
                elif event.button == 3:  # Правая кнопка мыши для удаления точек старта/цели
                    if current_mode == 'start_or_end':
                        remove_nearest_point(grid_x, grid_y)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            grid_x = x // cell_size
            grid_y = y // cell_size
            if is_within_bounds(grid_x, grid_y):
                if drawing:
                    if current_mode == 'road':
                        Pathes[grid_y, grid_x] = V_max  # Добавляем дорожку
                    elif current_mode == 'erase':
                        Pathes[grid_y, grid_x] = np.random.normal(loc=10, scale=1)  # Удаляем дорожку
                    elif current_mode == 'building':
                        Pathes[grid_y, grid_x] = BUILDING_VALUE  # Добавляем здание
                    elif current_mode == 'start_or_end':
                        add_unique_point(grid_y, grid_x)  # Добавляем точку старта или цели, если её ещё нет

    screen.fill(WHITE)
    draw_roads(Pathes)
    draw_grid()
    show_coordinates_on_screen(screen, (grid_y, grid_x))
    pygame.display.flip()

pygame.quit()

# Сохранение карты потенциалов и точек старта и цели

#Pathes[Pathes < 36.0] = 0
np.save(pathes_file, Pathes)
np.save('start_and_destination_points/' + start_and_destination_points_type + '.npy', np.array(start_and_destination_points))



