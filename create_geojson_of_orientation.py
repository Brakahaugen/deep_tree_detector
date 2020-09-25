import os
import csv
import json
import pyproj
from tqdm import tqdm

def appendToJsonFile(geometry_type, dir_id, img_number, lat, lon, height, omega, phi, kappa):
    jsonFile['features'].append({
        "type": "Feature",
        "geometry": {
            "type": geometry_type,
            "coordinates": [lat, lon, height]
        },
        "properties": {
            "dir_id": dir_id,
            "img_number": img_number,
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
        }
    })


topright = [63.4313291, 10.409034]
bottomleft = [63.422746, 10.3819406]



directory = "./Trondheim_imagery_exteriororientation"

jsonFile = {}
jsonFile['type'] = "FeatureCollection"
jsonFile['features'] = []

# appendToJsonFile("point", 10, 63, 100, 15800, 1, 80, 180, -90)

# wgs84=pyproj.CRS("EPSG:4326")
# utm=pyproj.CRS("EPSG:25832")
transformer = pyproj.Transformer.from_crs("epsg:25832", "epsg:4326")

for root, dirs, files in os.walk(directory):
    for file in tqdm(files):
        if file.endswith(".txt"):
            with open(directory + "/" + file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\n')
                next(csv_reader)
                for row in csv_reader:
                    elements = row[0].split(" ")
                    lat_lon = transformer.transform(elements[1], elements[2])       
                    # elements[1] = float(lat_lon[1])
                    # elements[2] = float(lat_lon[0])
                    # elements[3] = float(elements[3])
                    if topright[0] > lat_lon[0] and topright[1] > lat_lon[1] and bottomleft[0] < lat_lon[0] and bottomleft[1] < lat_lon[1]:
                        print(file)
                        print(lat_lon)

                    # appendToJsonFile("Point", file[:-4], *elements)


# with open('dataNoCOnv.geojson', 'w') as outfile:
#     json.dump(jsonFile, outfile)


