# Last inn ett stereo bilde
# Last inn posisjon og orientasjon for bilderetning
# Last inn trær som kan finnes på det samme sted
# Transformer treboks basert på avstand, vinkel fra kamera
# display treboks

#18115 is the folder of interest

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyproj
from tqdm import tqdm
import json
import math
import random
import numpy as np
from camera_params import camera_matrix, get_rotation_matrix, princ_offset_x, princ_offset_y, focal_dist, pix_size_x, pix_size_y


ORIGINAL_HEIGHT = 2672
ORIGINAL_WIDTH = 4008


def image_cords2viewbox(x,y):
    x = ORIGINAL_WIDTH - (x + ORIGINAL_WIDTH/2)
    y = (y + ORIGINAL_HEIGHT/2)
    return x,y

def get_pixel_coords_of_trees(trees: list, easting, northing, height, rotation_matrix):
    """
        input: list of trees
        tree: (east, north, height) aka (X,Y,Z)

        returns the projected pixel coordinate.
    """
    print("\n\n lets go")

    pixel_coord_list = []
    
    for tree in trees_inside:    
        print(tree)
        print(easting,northing, height)

        abs_tree = math.sqrt((tree[0] - easting)**2 + (tree[1] - northing)**2 + (tree[2] - height)**2)

        tree = np.array([[tree[0] - easting], [tree[1] - northing], [tree[2] - height]]) #Tree is now relative to camera, at origin.
        print("translated tree", tree)
        # [u, v, 1]^T = camera_matrix* [X,Y,Z]^T / Z_img
        tree = np.matmul(rotation_matrix,tree) #Tree is now rotated relative to the camera
        print("rotated tree",tree)
        if (tree[2,0] > 0) : continue
        tree = np.vstack([tree, 1])
        print("homoTree", tree)
        print("OKay dokay")
        pixel_coords = np.matmul(camera_matrix, tree) / (tree[2])
        print("pixel,coords", pixel_coords)
        print("shifted_coords", image_cords2viewbox(pixel_coords[0,0], pixel_coords[1,0]))
        pixel_coords = image_cords2viewbox(pixel_coords[0,0], pixel_coords[1,0])

        if pixel_coords[0] > 0 and pixel_coords[1] > 0:
            pixel_coord_list.append(pixel_coords)
            print("FOUND A GOOD POINT?\n\n\n\n\n\n\n")
            print(pixel_coords)
    return pixel_coord_list

def get_pixel_coords_of_trees_v2(trees: list, camera_world_coords: list, rotation_matrix):
    pixel_coord_list = []

    for tree in trees_inside: 
        # if is_behind_camera(tree, camera_world_coords, rotation_matrix): continue

        x_p = (princ_offset_x - focal_dist*(
            (rotation_matrix[0,0] *(tree[0] - camera_world_coords[0]) +
            rotation_matrix[1,0] *(tree[1] - camera_world_coords[1]) +
            rotation_matrix[2,0] *(tree[2] - camera_world_coords[2]))
            / 
            (rotation_matrix[0,2] *(tree[0] - camera_world_coords[0]) +
            rotation_matrix[1,2] *(tree[1] - camera_world_coords[1]) +
            rotation_matrix[2,2] *(tree[2] - camera_world_coords[2]))
            )) / pix_size_x
        
        y_p = (princ_offset_y - focal_dist*(
            (rotation_matrix[0,1] *(tree[0] - camera_world_coords[0]) +
            rotation_matrix[1,1] *(tree[1] - camera_world_coords[1]) +
            rotation_matrix[2,1] *(tree[2] - camera_world_coords[2]))
            / 
            (rotation_matrix[0,2] *(tree[0] - camera_world_coords[0]) +
            rotation_matrix[1,2] *(tree[1] - camera_world_coords[1]) +
            rotation_matrix[2,2] *(tree[2] - camera_world_coords[2]))
            )) / pix_size_y
            
        print("image_coords=", x_p, y_p)
        print("window_coords=", image_cords2viewbox(x_p,y_p))
        # print("is_behind_camera", is_behind_camera(tree, camera_world_coords, rotation_matrix))
        x_p, y_p = image_cords2viewbox(x_p,y_p)
        pixel_coord_list.append([x_p, y_p])
        (get_pixel_coords_of_trees_v3(trees_inside, camera_world_coords, rotation_matrix))

    return pixel_coord_list

def is_behind_camera(tree, camera_world_coords, rotation_matrix):
    tree = np.array([[tree[0] - camera_world_coords[0]], [tree[1] - camera_world_coords[1]], [tree[2] - camera_world_coords[2]]]) #Tree is now relative to camera, at origin.
    # print("translated tree", tree)
    tree = np.matmul(rotation_matrix,tree) #Tree is now rotated relative to the camera
    # print("rotated tree", tree)
    print(tree[2][0])
    return tree[2][0] > 0

def get_pixel_coords_of_trees_v3(trees: list, camera_world_coords: list, rotation_matrix):
    pixel_coord_list = []

    print(rotation_matrix)

    for tree in trees_inside:    

        tree = np.array([[tree[0] - easting], [tree[1] - northing], [tree[2] - height]]) #Tree is now relative to camera, at origin.
        print("translated tree", tree)
        tree = np.matmul(rotation_matrix,tree) #Tree is now rotated relative to the camera
        print("rotated tree", tree)
        pixel_coord_list.append(tree)
    
    return pixel_coord_list


def getImageDirection(omega: float, phi: float, kappa: float):
    """
        returns the image direction based on three orientation parameters
        omega is clockwise orientation around X
        phi is clockwise orientation around Y
        kappa is clockwise orientation around Z
    """

    return kappa

def checkPoint(radius, x, y, startAngle, endAngle): 
    """
        Checks if a point lies inside a circle sector. 
    """

    polarRadius = math.sqrt(x * x + y * y)  #In meters
    if polarRadius < radius:
        return polarRadius
    angle = math.degrees(math.atan2(x,y)) #In degrees
    widthPositionInPercent = abs((angle - startAngle)/(endAngle - startAngle)) #on what relative angle it is found
    if (polarRadius < radius):
        # and(angle >= startAngle and angle <= endAngle 
          
        print("Spotted a tree")
        print(angle, startAngle, endAngle)

        return [widthPositionInPercent, polarRadius]
    else: 
        return []
  

def find_trees_in_image(img_east_north: list, image_direction: float, image_direction_borders: list, max_distance: float):
    """
        find trees inside a imagerange
            img_lat_lon is a array with lat and lon floats
            image_direction is the direction of the image
            image_direction_borders holds the border of the image
    """
    reverse_transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:25832",)
    
    trees_in_pic = 0
    tree_values = []
    trees_for_advanced_proj = []

    for feature in (trees['features']):
        tree_lat_lon = (feature["geometry"]["coordinates"][0:3])
        tree_east_north_height = reverse_transformer.transform(tree_lat_lon[1], tree_lat_lon[0], tree_lat_lon[2])
        
        #[y-placement, depth]
        local_location = checkPoint(
            max_distance,
            tree_east_north_height[0] - img_east_north[0], #sets origin in image and x to be east-distance to tree
            tree_east_north_height[1] - img_east_north[1], #sets origin in image and y to be north-distance to tree
            *image_direction_borders) #start_angle and end_angle

        if local_location: #We found a tree in viewbox
            trees_in_pic += 1
            tree_values.append(local_location)
            trees_for_advanced_proj.append(tree_east_north_height)

    print("We found trees")
    print(trees_for_advanced_proj)
    return trees_for_advanced_proj
    print(f"trees in pic = {trees_in_pic}")
    return tree_values


def get_boxes(trees_inside: list, width: int, height: int):
    """
        given a list of treesposition and depth and image width and size, return an array of bboxes
    """
    
    width_factor = 2
    height_factor = 5.6
    

    bboxes = []
    for tree in trees_inside:
        print("tree value", tree)
        depth = tree[1]
        treeoffset = tree[0]
        
        x0 = width * treeoffset
        x1 = x0 - width/(width_factor * depth)
        if x1 > width:
            x1 = width
        if x1 < 0:
            x1 = 0

        y0 = (depth / height) * height * height_factor
        y1 = y0 + height*height_factor/depth
        if y1 > height:
            y1 = height

        bboxes.append([int(x0),int(y0),int(x1),int(y1)]) #x0,x1,y0,y1

    return bboxes


with open('data/fkb_naturinfo_posisjon_Featu.geojson') as f:
    trees = json.load(f)

transformer = pyproj.Transformer.from_crs("epsg:25832", "epsg:4326")
reverse_transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:25832",)

image_width = 65 #degrees
max_distance = 50 #meters

# folder = "18117"
folder = "18115"
img_path = "./frontSys/"+folder+"/"
meta_path = "./Trondheim_imagery_exteriororientation/"+folder+".txt"
l_r_path = [0,1]

images = {}

#imagenumber easting[m] northing[m] height[m] omega[degree] phi[degree] kappa[degree]
metadata = []

with open(meta_path) as file_in:
    next(file_in)
    for line in file_in:
        metadata.append(line.split(" "))

num_images = len(metadata)
if num_images >20:
    num_images = 20 #SUbject to change

for i in tqdm(range(num_images)):
    img_left = cv2.imread(img_path + str(l_r_path[0]) + "/" + str(i) + ".jpg") # reads image 'opencv-logo.png' as grayscale
    img_right = cv2.imread(img_path + str(l_r_path[1]) + "/" + str(i) + ".jpg") # reads image 'opencv-logo.png' as grayscale
    images[i] = (img_left, img_right, metadata[i])

index = 0
for image in images.items():

    easting = float(image[1][2][1])
    northing = float(image[1][2][2])
    height = float(image[1][2][3])
    omega, phi, kappa = float(image[1][2][4]), float(image[1][2][5]), float(image[1][2][6])


    #Tests:
    # rotation_matrix = get_rotation_matrix(omega,phi,kappa)
    # unit_vector = np.array([[0],[1],[0]])

    # print("unit_vector",unit_vector)
    # print("rotated", np.matmul(rotation_matrix, unit_vector))
    # break
    lat_lon = transformer.transform(easting, northing)       
    
    image_direction = getImageDirection(omega, phi, kappa) 
    image_direction_borders = [image_direction - image_width, image_direction + image_width]
    
    trees_inside = find_trees_in_image([easting,northing], image_direction, image_direction_borders, max_distance)


    print(trees_inside)
    # trees_inside(easting, northing - 30, height))
    
    print(height)
    
    
    #Test points for dir 18117 img 1
    # trees_inside = [(569284, 7034291, 16.8)]
    # trees_inside.append((569286, 7034279, 15.57))
    # trees_inside.append((569289, 7034267, 18.21))
    # trees_inside.append((569265, 7034280, 16.8))
    # trees_inside.append((569253, 7034317, 26.7))
    
    #Test points for dir 23396 img 1
    # trees_inside = [(569352, 7034238, 16.8)]
    # trees_inside.append((569343, 7034238, 16.57))
    # trees_inside.append((569333, 7034239, 16.21))
    # trees_inside.append((569325, 7034238, 16.1))
    # trees_inside.append((569315, 7034239, 16.7))
    
    # trees_inside = [(569352, 7034238, height)]
    # trees_inside.append((569343, 7034238, height))
    # trees_inside.append((569333, 7034239, height))
    # trees_inside.append((569325, 7034238, height))
    # trees_inside.append((569315, 7034239, height))
    
    
    # trees_inside.append([569354.00, 7034424.57, height])
    print(trees_inside)
    print(easting,northing,height)
    # trees_inside = 
    print(len(trees_inside))
    print(omega, phi, kappa)

    # bboxes = get_boxes(trees_inside, 960, 540)
    # print(bboxes)
    # imS = cv2.resize( image[1][1], (960, 540))
    #x0,x1,y0,y1
    # for bbox in bboxes:
    #     cv2.rectangle(imS, (bbox[2],bbox[3]), (bbox[0],bbox[1]), (255,255,255), 2) #(random.random()*255,random.random()*255,random.random()*255),2)

    # coords = get_pixel_coords_of_trees_v2(trees_inside, [easting, northing, height], get_rotation_matrix(omega, phi, kappa))

    print("coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(omega, phi, kappa))")
    coords = get_pixel_coords_of_trees_v2(trees, [easting, northing, height], get_rotation_matrix(omega, phi, kappa))
    print(coords)
    # coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(omega, phi, kappa))
    
    # print("coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(kappa, omega, phi))")
    # coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(kappa, omega, phi))
    
    # print("coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(phi, kappa.,omega, ))")
    # coords = get_pixel_coords_of_trees_v3(trees_inside, [easting, northing, height], get_rotation_matrix(phi, kappa, omega))
    indibindi = 0
    coords = get_pixel_coords_of_trees(trees_inside, easting, northing, height, get_rotation_matrix(omega, phi, kappa))
    for coord in coords:
            print("coord",coord)
            if coord[0] < ORIGINAL_WIDTH and coord[1] < ORIGINAL_HEIGHT and coord[0] >= 0 and coord[1] >= 0:
                print("\nFOUDN SOMETHING")
                # print(coords[indibindi])
                # print(coord)
                # print(trees_inside[indibindi])
                # print()
                

            indibindi = indibindi + 1
            cv2.circle(image[1][1], (int(coord[0]), int(coord[1])), 40, (0,255,0), 40)
    
    imS = cv2.resize(image[1][1], (int(ORIGINAL_WIDTH/4), int(ORIGINAL_HEIGHT/4)))

    cv2.imwrite("test_images/" + folder +str(index) + ".jpg",imS)
    index += 1

    coords = get_pixel_coords_of_trees(trees_inside, easting, northing, height, get_rotation_matrix(omega, phi, kappa))

    # coords = get_pixel_coords_of_trees_v2(trees_inside, [easting, northing, height], get_rotation_matrix(omega, phi, kappa))
    # for coord in coords:
    #     print("coord",coord)
    #     cv2.circle(imS, (coord[0]/0.009, coord[1]/0.009), 10, (0,0,255), 20)
    
    # print((coords[0]*960/2, coords[1]*540))

    cv2.imshow("output", imS)                            # Show image
    cv2.waitKey(0)
    