import boto3
from botocore.exceptions import NoCredentialsError

import os.path
from os import path

from tqdm import tqdm

aws_access_key_id="ASIA4TUSWSSFLNGEIPLO"
aws_secret_access_key="c6xC6ITmNl8RE/aFQLefPyptK5tXjgpomFIHtKyF"
aws_session_token="FwoGZXIvYXdzEC4aDK8oP0T1mheCkZ2gpCLDARhmRL8rS8SrPhsov9/Qjti5rqwARqDQrk1T3JCuE+mwKhLELGMhBA36+F05Nu2ylp7xEmoROxFGb5XKGsNnV4xIvLuH70ZIRfgMexLkQq2cFprrNH7vs3ZU/uoGpyGqSkSKzJb6s7h6EVP2PtEjECUslXRmlf4h3CFk5LzDPjY7oXxt+Z2+DngO9LX9/DaxtzHc+L8q2cPcANxl5P78wmviSpT1SZQbsGaBvqCz6SL+LNm8FCldmXNe96Yr+i7QyZlGsCi03Mz7BTItFznDZ+7VnEcZ11zKsfVDKl3PNGWKN4KMjwvhDzYI3dm2HGD5y++Jjxq7++PH"


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key, 
                      aws_session_token=aws_session_token)
    try:
        if path.exists(local_file):
            s3.upload_file(local_file, bucket, s3_file)
            # print("Upload Successful")
        else:
            s3.put_object(Body=local_file.encode('ascii'), Bucket=bucket, Key=s3_file)
            # print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


dir_entry = "frontSys/"
bucket_name = "trondhyme"
meta_path = "Trondheim_imagery_exteriororientation/"

# for i in range(16847,30565+1):
for i in tqdm(range(16848,16857)):
    
    metadata = []
    with open(meta_path + str(i) + ".txt") as file_in:
        next(file_in)
        for line in file_in:
            metadata.append(line.split(" "))
    
    l = os.listdir(dir_entry + str(i) + "/0/") # dir is your directory path
    for j in tqdm(range(len(l))):
        image_path = dir_entry + str(i) + "/0/" + str(j) + ".jpg"
        stereo_path = dir_entry + str(i) + "/1/" + str(j) + ".jpg"


        orientation = metadata[j]

        #Find boxes using orientation and stereo

        #Upload Image
        location = "images/train/"
        name = str(i) + "0" + str(j).zfill(3) + ".jpg"
        uploaded = upload_to_aws(image_path, bucket_name, location + name)
        assert(uploaded)

        #Upload stereoCounterpart
        location = "images/stereo_counterpart/"
        name = str(i) + "1" + str(j).zfill(3) + ".jpg"
        uploaded = upload_to_aws(image_path, bucket_name, location + name)
        assert(uploaded)     

        # #Upload orientation
        location = "images/orientation/"
        name = str(i) + "0" + str(j).zfill(3) + ".txt"
        stringified_list = str(metadata[j]).replace("[", "").replace("]", "").replace("'", "")
        uploaded = upload_to_aws(stringified_list, bucket_name, location + name)
        assert(uploaded)    

        #Upload bounding boxes ...