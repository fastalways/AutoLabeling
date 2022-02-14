import os, os.path
import re
# simple version for working with CWD
#print len([name for name in os.listdir('.') if os.path.isfile(name)])

# path joining version for other paths
DIR = 'D:/DatasetMedicalWaste/'
for name_folder in os.listdir(DIR):
    if os.path.isdir(os.path.join(DIR, name_folder)):
        for name in os.listdir(DIR+name_folder): ## in each folder
            print(f'Working in {name}')
            if os.path.isfile(os.path.join(DIR+name_folder, name)):
                full_filename = os.path.join(DIR+name_folder, name)
                filename, file_extension = os.path.splitext(full_filename)
                if(file_extension=='.txt'):
                    if os.path.exists(full_filename): # เช็คว่า path existed ?
                        with open(full_filename) as file:
                            lines = file.readlines()
                            lines = [line.rstrip() for line in lines]
                            xywh_str = re.split(r'\t+', lines[0])
                        if(len(xywh_str)==5):
                            write_text =  filename + '\t' + xywh_str[1] + '\t' + xywh_str[2] + '\t' + xywh_str[3] + '\t' + xywh_str[4]
                            f = open(full_filename, "w")
                            f.write(write_text)
                            f.close()
                        else:
                            print(f'txt pos error in {full_filename}')
print("Finished!!!!")
