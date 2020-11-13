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
import copy


ORIGINAL_HEIGHT = 2672
ORIGINAL_WIDTH = 4008
IMG_RES_SCALE = 1/4
TREE_HEIGHT_TO_WIDTH_RATIO = (1/4) / IMG_RES_SCALE
GLOBAL_ANNOTATION_ID = 0

def create_annotation(folder, index, x0,y0,x1,y1):
    global GLOBAL_ANNOTATION_ID

    annotation = {}
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["dir_id"] = folder
    annotation["image_id"] = index
    annotation["category_id"] = 1
    annotation["id"] = GLOBAL_ANNOTATION_ID
    annotation["bbox"] = [x0, y0, x1-x0, y1-y0] # (x, y, width, height)
    annotation["area"] = (x1-x0)*(y1-y0) #Should be for segmentation, but used for area of bbox
    GLOBAL_ANNOTATION_ID += 1   

    print(annotation)
    return annotation
    
    

    

def calculate_top_coords(bottom_coords: list, distance_to_tree: float):

    bottom_coords[0] - (ORIGINAL_HEIGHT/distance_to_tree)

    return bottom_coords

def get_pixel_coords_of_tree(tree: np.array, rotation_matrix):
    # print("translated tree", tree)

    tree = np.matmul(rotation_matrix,tree) #Tree is now rotated relative to the camera
    # print("rotated tree",tree)

    if (tree[2,0] > 0) : return None
    tree = np.vstack([tree, 1]) #Make tree homogenous
    
    pixel_coords = np.matmul(camera_matrix, tree) / (tree[2])
    print("pixel,coords", pixel_coords)
    print("shifted_coords", image_cords2viewbox(pixel_coords[0,0], pixel_coords[1,0]))
    return image_cords2viewbox(pixel_coords[0,0], pixel_coords[1,0])


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

    pixel_coord_list = []
    
    for tree_feature in trees_inside:    

        tree = tree_feature["geometry"]["coordinates"]
        print(tree)

        if tree_feature["properties"]["hoydereferanse"] == "TOP":
            tree_feature["tree_top_pixel_coords"] = get_pixel_coords_of_tree(np.array([[tree[0] - easting], [tree[1] - northing], [tree[2] - height]]), rotation_matrix)
        tree_feature["tree_bottom_pixel_coords"] = get_pixel_coords_of_tree(np.array([[tree[0] - easting], [tree[1] - northing], [tree_feature["properties"]["dtm10_hoyde"] - height]]), rotation_matrix)
        tree_feature["distance_to_tree"] = ((tree[0] - easting)**2 + (tree[1] - northing)**2)**(0.5)
        pixel_coord_list.append(tree_feature)
    return pixel_coord_list

def clip_box_against_window(x0,y0,x1,y1):

    if x0 > ORIGINAL_WIDTH or y0 > ORIGINAL_HEIGHT: return None,None,None,None
    if x1 < 0 or y1 < 0: return None,None,None,None
    
    x0 = x0 if x0 > 0 else 0
    y0 = y0 if y0 > 0 else 0
    x1 = x1 if x1 < ORIGINAL_WIDTH else ORIGINAL_WIDTH
    y1 = y1 if y1 < ORIGINAL_HEIGHT else ORIGINAL_HEIGHT

    return x0,y0,x1,y1

def create_box(bottom_coords, top_coords):
    width = (bottom_coords[1] - top_coords[1]) * TREE_HEIGHT_TO_WIDTH_RATIO
    print("width",width)
    print("width",width)
    print("width",width)
    print("width",width)
    print("width",width)
    
    x0,y0,x1,y1 = (top_coords[0] - width/2), top_coords[1],( bottom_coords[0] + width/2), (bottom_coords[1])

    return clip_box_against_window(x0,y0,x1,y1)

def checkPoint(radius, x, y):
    """
        Checks if a point lies inside a circle of the given camera_location. 
    """
    polarRadius = math.sqrt(x * x + y * y)  #In meters
    if polarRadius < radius:
        return polarRadius
  

def find_trees_in_image(img_east_north: list, max_distance: float, trees):
    """
        find trees inside a imagerange
            img_lat_lon is a array with lat and lon floats 
            image_direction is the direction of the image
            image_direction_borders holds the border of the image
    """
    reverse_transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:25832",)
    
    trees_for_advanced_proj = []

    for feature in (trees['features']):
        tree_lat_lon = (feature["geometry"]["coordinates"][0:3])
        tree_east_north_height = reverse_transformer.transform(tree_lat_lon[1], tree_lat_lon[0], tree_lat_lon[2])
        feature["geometry"]["coordinates"] = tree_east_north_height
        #[y-placement, depth]
        is_inside = checkPoint(
            max_distance,
            tree_east_north_height[0] - img_east_north[0], #sets origin in image and x to be east-distance to tree
            tree_east_north_height[1] - img_east_north[1]) #sets origin in image and y to be north-distance to tree

        if is_inside: #We found a tree in viewbox
            trees_for_advanced_proj.append(feature)

    print("We found trees")
    print(trees_for_advanced_proj)
    return trees_for_advanced_proj


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


with open('data/trees_with_dtm10_height.geojson') as f:
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

annotations = []

images = {}

#imagenumber easting[m] northing[m] height[m] omega[degree] phi[degree] kappa[degree]
metadata = []

with open(meta_path) as file_in:
    next(file_in)
    for line in file_in:
        metadata.append(line.split(" "))

num_images = len(metadata)
# num_images = 1
if num_images >20:
    num_images = 20 #SUbject to change

for i in tqdm(range(num_images)):
    img_left = cv2.imread(img_path + str(l_r_path[0]) + "/" + str(i) + ".jpg") # reads image 'opencv-logo.png' as grayscale
    img_right = cv2.imread(img_path + str(l_r_path[1]) + "/" + str(i) + ".jpg") # reads image 'opencv-logo.png' as grayscale
    images[i] = (img_left, img_right, metadata[i])

index = 0
for image in images.items():

    easting, northing, height = float(image[1][2][1]), float(image[1][2][2]), float(image[1][2][3])
    omega, phi, kappa = float(image[1][2][4]), float(image[1][2][5]), float(image[1][2][6])

    trees_inside = find_trees_in_image([easting,northing], max_distance, copy.deepcopy(trees))

    tree_features = get_pixel_coords_of_trees(trees_inside, easting, northing, height, get_rotation_matrix(omega, phi, kappa))

    for tree in tree_features:
        if tree["tree_bottom_pixel_coords"] == None: 
            continue
        
        bottom_coords = tree["tree_bottom_pixel_coords"]
        top_coords = tree["tree_top_pixel_coords"] if tree["tree_top_pixel_coords"] != None else calculate_top_coords(bottom_coords, tree["distance_to_tree"])
        print("bottom_coords",bottom_coords)
        print("top_coords",top_coords)
        # for coord in [bottom_coords, top_coords]:
        #     cv2.circle(image[1][1], (int(coord[0]), int(coord[1])), 40, (0,255,0), 40)

        x0,y0,x1,y1 = create_box(bottom_coords, top_coords)
        print(x0,y0,x1,y1)
        if x0 is None: continue
        cv2.rectangle(image[1][1], (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 20) 
        # cv2.rectangle(image[1][1],   (0,0),(800,900),(0,0,0),30)
        annotations.append(create_annotation(folder, index, x0,y0,x1,y1))



    imS = cv2.resize(image[1][1], (int(ORIGINAL_WIDTH*IMG_RES_SCALE), int(ORIGINAL_HEIGHT*IMG_RES_SCALE)))

    cv2.imwrite("test_images/" + folder +str(index) + ".jpg",imS)
    index += 1


    cv2.imshow("output", imS)                            # Show image
    cv2.waitKey(0)
    