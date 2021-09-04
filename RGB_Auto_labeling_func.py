import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import copy

img_path = './Dataset Medical Waste/3Waystopcock/'

diff_thres_r = 50  #0-359
diff_thres_g = 50  #0-255 180
diff_thres_b = 50   #0-255 150
alpha_value = .5 # 0.1-1

def pointBG(src_img):
    global diff_thres_r,diff_thres_g,diff_thres_b
    list_rists = []
    img = src_img.copy()
    # calc HSV hist
    for i in range(0,3):
        if(i==0): #H
            histr = cv.calcHist(img,[i],None,[256],[0,256])
        elif(i==1): #S
            histr = cv.calcHist(img,[i],None,[256],[0,256])
        elif(i==2): #V
            histr = cv.calcHist(img,[i],None,[256],[0,256])
        list_rists.append(histr)
    list_rists_np = np.array(list_rists,dtype=object)
    max_r = np.unravel_index(np.argmax(list_rists_np[0], axis=None), list_rists_np[0].shape)
    max_g = np.unravel_index(np.argmax(list_rists_np[1], axis=None), list_rists_np[1].shape)
    max_b = np.unravel_index(np.argmax(list_rists_np[2], axis=None), list_rists_np[2].shape)
    print(f"(max_r={max_r},max_g={max_g},max_b={max_b}")
    low_r = np.int16(np.clip(max_r[0] - diff_thres_r,0,359)).item()
    low_g = np.int16(np.clip(max_g[0] - diff_thres_g,0,255)).item()
    low_b = np.int16(np.clip(max_b[0] - diff_thres_b,0,255)).item()
    high_r = np.int16(max_r[0] + diff_thres_r).item()
    high_g = np.int16(max_g[0] + diff_thres_g).item()
    high_b = np.int16(max_b[0] + diff_thres_b).item()
    lowerb = (low_r, low_g, low_b)
    upperb = (high_r, high_g, high_b)
    lowerb = (low_r, low_g, low_b)
    upperb = (high_r, high_g, high_b)
    print(f"lowerb{lowerb}")
    print(f"upperb{upperb}")
    threshold_img = cv.inRange(img, lowerb, upperb)
    return threshold_img


def main():
    global img_path,alpha_balue
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
        imgs.append(cv.resize(tmp_img,(w,h)))
    # Set low contrast
    lowct_imgs = []
    for i,img in enumerate(imgs):
        lowct_imgs.append(cv.convertScaleAbs(img,alpha=alpha_value, beta=0))
    # Convert to RBG
    RGB_imgs = []
    for i,img in enumerate(lowct_imgs):
        RGB_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # Call func pointBG
    pointBG_imgs = []
    for i,img in enumerate(RGB_imgs):
        pointBG_imgs.append(pointBG(img))
    
    # Display by plt
    plt_index = 1
    num_imgs = len(imgs)
    col = 2
    for i in range(num_imgs):
        if i==1 :
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.title("Original"),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.title("LowContrast"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.title("HSV"),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(pointBG_imgs[i]),plt.title("pointBG"),plt.xticks([]),plt.yticks([])
            plt_index+=1
        else :
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(pointBG_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
    plt.show()




if __name__ == "__main__":
    main()