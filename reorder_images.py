import os
from PIL import Image
import time
import csv





def one_iteration(images_list: list, x):
    return_list = images_list.copy()

    d_orientation = 0
    last_ori = None

    for i in images_list:
        # print(i%2)
        # if i%2 > 0:
        #     continue
        

        first = True
        first_first = True

        ori_path = "./orientation/" + str(i) + ".txt"
        with open(ori_path, newline='') as csvfile:
            orientation_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in orientation_reader:
                if first:
                    if first_first:
                        first_first = False
                        continue
                    start_ori = row[1:4]
                    first = False
                    if last_ori is not None:
                        d_orientation = (
                            (float(start_ori[0]) - float(last_ori[0]))**2 +
                            (float(start_ori[1]) - float(last_ori[1]))**2 +
                            (float(start_ori[2]) - float(last_ori[2]))**2
                        )**(0.5)

                        print(d_orientation)
                last_ori_candidate = row[1:4]

        
        if d_orientation > 15:
            continue

        last_ori = last_ori_candidate
        x += 1
        return_list.remove(i)





        for j in range(0,1):
            folder = "./images/" + str(i) + "/" + str(j) + "/"
            
            k = 0

            while True:
                file = folder + str(k) + ".jpg"
                try:
                    im = Image.open(file)
                except:
                    break
                print(file)
                if k < 10:
                    k_ = str("00" + str(k))
                elif k < 100:
                    k_ = str(0) + str(k) 
                print("./" + str(j) + "/" + str(i) + str(j) + str(k_) + ".jpg")
                im = im.save("./" + str(j) + "/" + str(x) + str(j) + str(k_) + ".jpg") 
                k += 1
    return return_list

# for root, dirs, files in os.walk("./images", topdown=False):
#     print(root)

#     time.sleep(1)
#     for dir in dirs:
     
    
x = 0
images_list = list(range(85300, 85442))
print(images_list)

while len(images_list) > 0:
    for i in images_list:
        print(i)
    # print(images_list)
    images_list = one_iteration(images_list, x)

