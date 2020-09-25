import os
from PIL import Image
import time
import csv






def find_one(images_list: list, start_i: int, chain: list):

    ori_path = "./orientation/" + str(start_i) + ".txt"
    with open(ori_path, newline='') as csvfile:
            orientation_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in orientation_reader:
                last_ori = row[1:4]

    for i in images_list:
            # print(i%2)
            # if i%2 > 0:
            #     continue
        
        first = True
        firstfirst = True
        ori_path = "./orientation/" + str(i) + ".txt"
        with open(ori_path, newline='') as csvfile:
            orientation_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in orientation_reader:
                if first:
                    if firstfirst:
                        firstfirst = False
                        continue
                    start_ori = row[1:4]
                    first = False
                    if last_ori is not None:
                        d_orientation = (
                            (float(start_ori[0]) - float(last_ori[0]))**2 +
                            (float(start_ori[1]) - float(last_ori[1]))**2 +
                            (float(start_ori[2]) - float(last_ori[2]))**2
                        )**(0.5)

                        # print(d_orientation)

        
        if d_orientation > 15:
            continue

        images_list.remove(i)
        chain.append(i)
        return find_one(images_list, i, chain)
    return images_list, i, chain

best_chain = []
for i in range(85300, 85442):
    images_list = list(range(85300, 85442))
    chain = [i]
    images_list, start_i, length = find_one(images_list, i, chain)

    if len(chain) > len(best_chain):
        best_chain = list(chain)
    print(start_i)
    print(chain)

images_list = list(range(85300, 85442))
print(best_chain)
print(len(best_chain))
print(len(images_list))
for x in best_chain:
    print(x)
    images_list.remove(x)
print(len(images_list))