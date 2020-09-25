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

def checkPoint(radius, x, y, startAngle, endAngle): 
    # Python3 program to check if a point  
    # lies inside a circle sector. 

      # Calculate polar co-ordinates 
    polarradius = math.sqrt(x * x + y * y) 
    Angle = math.degrees(math.atan2(x,y)) 
    # Angle = np.rad2deg(Angle)%360 - 180
    # print(Angle)
    # print(x,y)
    
    total_angle = endAngle - startAngle
    offset = Angle - startAngle

    # Check whether polarradius is less 
    # then radius of circle or not and  
    # Angle is between startAngle and  
    # endAngle or not 
    if (Angle >= startAngle and Angle <= endAngle 
        and polarradius < radius): 
        print(offset/total_angle)
        print("offset-^")
        return [offset/total_angle, polarradius]
    else: 
        return False 
  

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

    for feature in (trees['features']):
        tree_lat_lon = (feature["geometry"]["coordinates"][0:2])
        tree_east_north = reverse_transformer.transform(tree_lat_lon[1], tree_lat_lon[0])
        
        #[y-cordinate, depth]
        local_location = checkPoint(max_distance, tree_east_north[0] - img_east_north[0], tree_east_north[1] - img_east_north[1], image_direction_borders[0], image_direction_borders[1])
        if local_location:
            print("tree is actually inside")
            print(feature["geometry"]["coordinates"])
            print("tree, img")
            print(tree_east_north)
            print(img_east_north)
            print(local_location)
            print(f"image direction = {image_direction}")
            # print(local_location)
            # Nå må det lages en bounding box for treet

            trees_in_pic += 1
            tree_values.append(local_location)


    print(f"trees in pic = {trees_in_pic}")

    return tree_values


def get_boxes(trees_inside: list, width: int, height: int):
    """
        given a list of treesposition and depth and image width and size, return an array of bboxes
    """
    width_factor = 25
    height_factor = 5
    

    bboxes = []
    for tree in trees_inside:
        depth = tree[1]
        treeoffset = tree[0]
        
        x0 = width * treeoffset
        x1 = x0 + width/(width_factor * depth)
        if x1 > width:
            x1 = width

        y0 = (depth / height) * height * height_factor
        y1 = y0 + height*height_factor/depth
        if y1 > height:
            y1 = height

        bboxes.append([int(x0),int(y0),int(x1),int(y1)]) #x0,x1,y0,y1

    return bboxes

with open('fkb_naturinfo_posisjon_Featu.geojson') as f:
    trees = json.load(f)

transformer = pyproj.Transformer.from_crs("epsg:25832", "epsg:4326")
reverse_transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:25832",)

image_width = 40 #degrees
max_distance = 30 #meters

img_path = "./frontSys/18115/"
meta_path = "./Trondheim_imagery_exteriororientation/18115.txt"
l_r_path = [0,1]
num_images = 9

images = {}

#imagenumber easting[m] northing[m] height[m] omega[degree] phi[degree] kappa[degree]
metadata = []

with open(meta_path) as file_in:
    next(file_in)
    for line in file_in:
        print(line.split(" "))
        metadata.append(line.split(" "))
print(metadata)


for i in tqdm(range(num_images)):
    img_left = cv2.imread(img_path + str(l_r_path[0]) + "/" + str(i) + ".jpg",0) # reads image 'opencv-logo.png' as grayscale
    img_right = cv2.imread(img_path + str(l_r_path[1]) + "/" + str(i) + ".jpg",0) # reads image 'opencv-logo.png' as grayscale
    images[i] = (img_left, img_right, metadata[i])



for image in images.items():

    easting = float(image[1][2][1])
    northing = float(image[1][2][2])
    kappa = float(image[1][2][6])
    

    lat_lon = transformer.transform(easting, northing)       
    # lat_lon = [lat_lon[1], lat_lon[0]] #reverse
    # print(lat_lon)
    image_direction = kappa # This is a coarse assumption, fix this later
    
    image_direction_borders = [kappa - image_width, kappa + image_width]

    # print(image_direction)
    # print(image_direction_borders)
    trees_inside = find_trees_in_image([easting,northing], image_direction, image_direction_borders, max_distance)

    print(trees_inside)
    bboxes = get_boxes(trees_inside, 960, 540)
    print(bboxes)
    print(bboxes[0][0])
    print(bboxes[0][0])
    print(bboxes[0][0])
    # break
    imS = cv2.resize( image[1][1], (960, 540))
    #x0,x1,y0,y1
    cv2.rectangle(imS, (bboxes[0][2],bboxes[0][3]), (bboxes[0][0],bboxes[0][1]), (0,0,255),2)
    cv2.imshow("output", imS)                            # Show image
    cv2.waitKey(0)