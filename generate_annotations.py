
import os
import pandas as pd
from tqdm import tqdm
import json
from projection import run_projection_method_on_that_image
from tqdm import tqdm

min_dir = 16847 
max_dir = 30565
path = "./data/images_60m_from_tree.json"

#Get all files in the directory

count = 0
annotations = []
with open(path) as json_file: 
    data = json.load(json_file) 

    for line in tqdm(data["features"]):
        feature = line["attributes"]
        
        if int(feature["dir_id"]) >= min_dir and int(feature["dir_id"]) <= max_dir:
            # print(int(feature["dir_id"]), int(feature["img_number"]))
            # print()
            count +=1
            annotation = run_projection_method_on_that_image(feature["dir_id"], feature["img_number"])
            if annotation:
                annotations.extend(annotation)
        
        if count > 100:
            break
            
print(count)

with open('data/annotations.json', 'w') as outfile:
    json.dump(annotations, outfile,indent=4)

# secondaryList["dtmhoyde"]