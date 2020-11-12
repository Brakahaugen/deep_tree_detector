import os
import shutil
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
from tqdm import tqdm



# swi = open(r'\\ibmrs01.ibm.ntnu.no\storheia\Trondheimdata\HDD_1\Trondheim_imagery\16847\0\0.jpg')
# print(swi)
# path_string = "//ibmrs01.ibm.ntnu.no/storheia/Trondheimdata/HDD_1/Trondheim_imagery/16850/0/0.jpg"
# //HOST/share/path/to/file

# i = 1

# f = open(r"\ibmrs01.ibm.ntnu.no\storheia\Trondheimdata\fkb_naturinfo_posisjon_Featu.geojson")
        # r"\DriveName\then\file\path\txt.md"
        
# im = Image.open(path_string)
# im.show()

dir_entry = "//ibmrs01.ibm.ntnu.no/storheia/Trondheimdata/HDD_1/Trondheim_imagery/"

# for i in range(16847,30565+1):
for i in tqdm(range(16845,16846)):

    l = os.listdir(dir_entry + str(i) + "/1/") # dir is your directory path
    for j in range(len(l)):
        try:
            full_path = dir_entry + str(i) + "/1/" + str(j) + ".jpg"
            target_name = str(i) + "1" + str(j).zfill(3) + ".jpg"
            shutil.move(full_path, dir_entry + target_name)
            print("Succesfully moved image")
        except:
            print("Something unexpected happened")
            break