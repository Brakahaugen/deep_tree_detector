from PIL import Image
from io import BytesIO
import numpy as np
import boto3
from tqdm import tqdm



aws_access_key_id="ASIA4TUSWSSFLNGEIPLO"
aws_secret_access_key="c6xC6ITmNl8RE/aFQLefPyptK5tXjgpomFIHtKyF"
aws_session_token="FwoGZXIvYXdzEC4aDK8oP0T1mheCkZ2gpCLDARhmRL8rS8SrPhsov9/Qjti5rqwARqDQrk1T3JCuE+mwKhLELGMhBA36+F05Nu2ylp7xEmoROxFGb5XKGsNnV4xIvLuH70ZIRfgMexLkQq2cFprrNH7vs3ZU/uoGpyGqSkSKzJb6s7h6EVP2PtEjECUslXRmlf4h3CFk5LzDPjY7oXxt+Z2+DngO9LX9/DaxtzHc+L8q2cPcANxl5P78wmviSpT1SZQbsGaBvqCz6SL+LNm8FCldmXNe96Yr+i7QyZlGsCi03Mz7BTItFznDZ+7VnEcZ11zKsfVDKl3PNGWKN4KMjwvhDzYI3dm2HGD5y++Jjxq7++PH"

session = boto3.Session(aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key, 
                      aws_session_token=aws_session_token)

key = "images/train/168470015.jpg"

s3 = session.resource('s3')
bucket = s3.Bucket("trondhyme")

for i in tqdm(range(34)):
    key = "images/train/168470" + str(i).zfill(3) + ".jpg"

    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
