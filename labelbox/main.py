import os
from shutil import copyfile
# swi = open(r'\\ibmrs01.ibm.ntnu.no\storheia\Trondheimdata\HDD_1\Trondheim_imagery\16847\0\0.jpg')
# print(swi)

i = 1
folder = 16847
while(folder < 16850):
    while(True):
        try:
            path = r'\\ibmrs01.ibm.ntnu.no\storheia\Trondheimdata\HDD_1\Trondheim_imagery\{}\0\{}.jpg' .format(folder, i)
            swi = open(path)
            copyfile((path), "./")

            print(swi)      
            i += 1  
        except:
            print("ngi")
            break
    folder += 1
    i = 0

