import numpy as np



class VisionKernel:
    __slots__ = ("dy", "dx", "w")
    def __init__(self, dy, dx, w):
        self.dy = dy.astype(np.int32)
        self.dx = dx.astype(np.int32)
        self.w  = w.astype(np.float32)

def precompute_kernels(d, c):
    
    """
    Returns list of 8 VisionKernel.
    """
    
    kernels = []


    for dir_idx in range(8):
        
        alpha = dir_idx * np.pi / 4
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)

        # local coordinates:
        
        forwards  = np.arange(0, d + 1)
        laterals  = np.arange(-d, d + 1)
        F, L      = np.meshgrid(forwards, laterals, indexing="ij")    

        mask = np.abs(L) <= F / 2
        Fm   = F[mask].astype(float)
        Lm   = L[mask].astype(float)

        # distances in local coordinate system:
        
        dist = np.hypot(Fm, Lm)

        # rotation:
        
        dy =  Fm * cos_a + Lm * sin_a
        dx = -Fm * sin_a + Lm * cos_a
        dy = np.rint(dy).astype(int)
        dx = np.rint(dx).astype(int)

        w = np.exp(-c * dist)

        kernels.append(VisionKernel(dy, dx, w))

    return kernels



