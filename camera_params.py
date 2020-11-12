import numpy as np
from math import cos, sin, radians, pi
#ALL IN M
focal_dist = 21.7258*10**(-3)
# focal_dist = 35*10**(-3)

princ_offset_x = -0.4446*10**(-3)
princ_offset_y = 0.2109*10**(-3)
# princ_offset_x = princ_offset_y = 0 

pix_size_x = 0.009*10**(-3)
pix_size_y = 0.009*10**(-3)

skew = 0

camera_matrix = np.array(
[
    [focal_dist/pix_size_x,     skew,                   princ_offset_x,     0  ],
    [0,                         focal_dist/pix_size_y,  princ_offset_y,     0  ],
    [0,                         0,                      1,                  0  ],
]) 


def get_rotation_matrix(omg, phi, kap):
    omg = radians(omg)
    phi = radians(phi)
    kap = radians(kap)
    
    return np.array(
    [
        [cos(phi)*cos(kap),      sin(omg)*sin(phi)*cos(kap) + cos(omg)*sin(kap),   -cos(omg)*sin(phi)*cos(kap) + sin(omg)*sin(kap)  ],
        [-cos(phi)*sin(kap),    -sin(omg)*sin(phi)*sin(kap) + cos(omg)*cos(kap),    cos(omg)*sin(phi)*sin(kap) + sin(omg)*cos(kap)  ],
        [sin(phi),              -sin(omg)*cos(phi),                                 cos(omg)*cos(phi)                               ],
    ]) 

def get_translation_matrix(easting,northing,height):
    
    translation_matrix = np.zeros([4,4])
    translation_matrix[0,0] = translation_matrix[1,1] = translation_matrix[2,2] = translation_matrix[3,3] = 1
    translation_matrix[0,3] = -easting
    translation_matrix[1,3] = -northing
    translation_matrix[2,3] = -height
    return translation_matrix



# TO use_
# Z[u, v, 1]^T = camera_matrix* [X,Y,Z]^T