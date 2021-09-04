import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import copy

img_path = './Dataset Medical Waste/Mask/'

diff_thres_h = 70  #0-359
diff_thres_s = 200  #0-255 
diff_thres_v = 120   #0-255 
alpha_value = .7 # 0.1-1

def pointBG(src_img):
    global diff_thres_h,diff_thres_s,diff_thres_v
    list_hists = []
    img = src_img.copy()
    # calc HSV hist
    for i in range(0,3):
        if(i==0): #H
            histr = cv.calcHist(img,[i],None,[360],[0,360])
        elif(i==1): #S
            histr = cv.calcHist(img,[i],None,[256],[0,256])
        elif(i==2): #V
            histr = cv.calcHist(img,[i],None,[256],[0,256])
        list_hists.append(histr)
    list_hists_np = np.array(list_hists,dtype=object)
    max_h = np.unravel_index(np.argmax(list_hists_np[0], axis=None), list_hists_np[0].shape)
    max_s = np.unravel_index(np.argmax(list_hists_np[1], axis=None), list_hists_np[1].shape)
    max_v = np.unravel_index(np.argmax(list_hists_np[2], axis=None), list_hists_np[2].shape)
    print(f"(max_h={max_h},max_s={max_s},max_v={max_v}")
    low_H = np.int16(np.clip(max_h[0] - diff_thres_h,0,359)).item()
    low_S = np.int16(np.clip(max_s[0] - diff_thres_s,0,255)).item()
    low_V = np.int16(np.clip(max_v[0] - diff_thres_v,0,255)).item()
    high_H = np.int16(max_h[0] + diff_thres_h).item()
    high_S = np.int16(max_s[0] + diff_thres_s).item()
    high_V = np.int16(max_v[0] + diff_thres_v).item()
    lowerb = (low_H, low_S, low_V)
    upperb = (high_H, high_S, high_V)
    lowerb = (low_H, low_S, low_V)
    upperb = (high_H, high_S, high_V)
    print(f"lowerb{lowerb}")
    print(f"upperb{upperb}")
    threshold_img = cv.inRange(img, lowerb, upperb)
    return threshold_img


def main():
    global img_path,alpha_value
    list_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    del_lists = []
    for i,fname in enumerate(list_files):
        last = len(fname) - 1
        file_ext = fname[-3:]
        if(file_ext!='jpg'): # and file_ext!='JPG'
            del_lists.append(fname) # mark as delete
            #print(file_ext)
    for val in del_lists:
        list_files.remove(val)
            
    print(f"After del other file ext:{list_files}")
    imgs = []
    #  ,
    # Read images from lists
    for i,fname in enumerate(list_files):
        tmp_img = cv.imread(img_path+fname)
        w = tmp_img.shape[0]//8
        h = tmp_img.shape[1]//8
        imgs.append(cv.resize(tmp_img,(h,w)))
    # Set low contrast
    lowct_imgs = []
    for i,img in enumerate(imgs):
        lowct_imgs.append(cv.convertScaleAbs(img,alpha=alpha_value, beta=0))
    # Convert to HSV
    HSV_imgs = []
    for i,img in enumerate(lowct_imgs):
        HSV_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL))
    # Call func pointBG
    pointBG_imgs = []
    for i,img in enumerate(HSV_imgs):
        pointBG_imgs.append(pointBG(img))
    
    # Display by plt
    plt_index = 1
    num_imgs = len(imgs)
    col = 3
    for i in range(num_imgs):
        if i==1 :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.title("Original"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.title("LowContrast"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.title("HSV"),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(pointBG_imgs[i]),plt.title("pointBG"),plt.xticks([]),plt.yticks([])
            plt_index+=1
        else :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(pointBG_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
    plt.show()




if __name__ == "__main__":
    main()