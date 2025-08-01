import cv2, numpy as np, matplotlib.pyplot as plt




def image_to_potential(fname_img,
                       N_x, 
                       N_y,
                       V_max,
                       invert = True,
                       show = True):
    """
    • fname_img – путь к исходному PNG/JPG.
    • invert=True  : более светлое (=белое) считается высокой протоптанностью.
                     Если у вас уже белое = фон, передайте  --no-invert .
    """
    # 1) image loading and grayscaling:
    
    img = cv2.imread(str(fname_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(fname_img)

    # 2) size changing:
    
    img = cv2.resize(img, (N_x, N_y), interpolation=cv2.INTER_AREA)

    # 3) iverting:
    
    if invert:
        img = 255 - img

    # 4) normalization:
    
    img_f = img.astype(np.float32)
    img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-6)
    
    V_target = img_f * V_max        
    V_target[V_target < 1] = V_max
    # 5) saving:
    np.save("target_map_square.npy", V_target)
    print(f"[*] Saved target_V.npy  (shape {V_target.shape}, dtype float32)")

    
    if show:
        plt.imshow(V_target, cmap="hot"); plt.colorbar(); plt.title("target_V")
        plt.show()

    return V_target



N_x, N_y, V_max, gaussian_initial_map = np.load('grid_params.npy')
N_x = int(N_x)
N_y = int(N_y)
V_max = int(V_max)

fname_img = 'helbing_square.png'

# Target map creating:

image_to_potential(fname_img = fname_img, N_x = N_x, N_y = N_y, V_max = V_max, invert = False,show = True)

V_target = np.load('target_map_square.npy')
plt.imshow(V_target, cmap="hot"); plt.colorbar(); plt.title("target_map")
plt.show()

