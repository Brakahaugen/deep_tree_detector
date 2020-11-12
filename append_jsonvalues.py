import os
import pandas as pd
from tqdm import tqdm
base_dir = 'data/'

secondary = "data/trees_with_dtm10hoyde.geojsonl.json"
primary = "data/fkb_naturinfo_posisjon_Featu.geojson"

#Get all files in the directory

primaryList = []
secondaryList = []

json_data = pd.read_json(secondary, lines=True)
secondaryList.append(json_data)

id_to_height_dict = {}
for line in secondaryList[0]["properties"]:
    print()
    id_to_height_dict[line["OBJECTID"]] = line["dtmhoyde"]
    print(line["OBJECTID"])


  
import json

def write_json(data, filename='data/trees_with_dtm10_height.geojson'): 
    with open(filename,'w+') as f: 
        json.dump(data, f, indent=4) 

with open(primary) as json_file: 
    data = json.load(json_file) 
      
    # temp = data['emp_details'] 
  
    # python object to be appended 
    y = {"emp_name":'Nikhil', 
         "email": "nikhil@geeksforgeeks.org", 
         "job_profile": "Full Time"
        } 

for f in tqdm(data["features"]):
    # print(f)
    # print(f["properties"]["OBJECTID"])
    f["properties"]['dtm10_hoyde'] = id_to_height_dict[f["properties"]["OBJECTID"]]
    # break
    # appending data to emp_details  
    # temp.append(y) 
      
write_json(data)  


# secondaryList["dtmhoyde"]