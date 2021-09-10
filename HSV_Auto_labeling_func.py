import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import copy

img_path = './Dataset Medical Waste/black/'

alpha_value = .7 # 0.1-1

mean_black = np.array([26,11,53])
mean_white = np.array([50,13,117])
mean_green_cam = np.array([110,191,122])
mean_green_mobile = np.array([119,191,101])

diff_thres_black = np.array([50,80,100]) # [70,110,40]  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_white = np.array([70,120,50])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_green_cam = np.array([70,110,40])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_green_mobile = np.array([70,120,50])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def pointBG(src_img):
    global mean_black,mean_white,mean_green_cam,mean_green_mobile
    global diff_thres_black,diff_thres_white,diff_thres_cam,diff_thres_mobile
    list_hists = []
    img = src_img.copy()
    # calc HSV hist
    hsv_planes = cv.split(img)
    for i in range(0,3):
        if(i==0): #H
            histr = cv.calcHist(hsv_planes,[i],None,[360],[0,360])
        elif(i==1): #S
            histr = cv.calcHist(hsv_planes,[i],None,[256],[0,256])
        elif(i==2): #V
            histr = cv.calcHist(hsv_planes,[i],None,[256],[0,256])
        list_hists.append(histr)
    list_hists_np = np.array(list_hists,dtype=object)
    max_h = np.unravel_index(np.argmax(list_hists_np[0], axis=None), list_hists_np[0].shape)
    max_s = np.unravel_index(np.argmax(list_hists_np[1], axis=None), list_hists_np[1].shape)
    max_v = np.unravel_index(np.argmax(list_hists_np[2], axis=None), list_hists_np[2].shape)
    #print(f"(max_h={max_h},max_s={max_s},max_v={max_v}")
    #print(f"{max_h[0]}\t{max_s[0]}\t{max_v[0]}")
    peak_HSV = np.array([max_h[0],max_s[0],max_v[0]])
    #print(peak_HSV)
    # หาค่าความแตกต่างระหว่างสีของภาพ และค่าเฉลี่ยพื้นหลังของแต่ละสี
    diffBG = np.array([sum(abs(mean_black-peak_HSV)),sum(abs(mean_white-peak_HSV)),sum(abs(mean_green_cam-peak_HSV)),sum(abs(mean_green_mobile-peak_HSV))])
    idxMatchedBG = np.unravel_index(np.argmin(diffBG, axis=None), diffBG.shape)[0]
    #dict_bg = {0:"Black",1:"White",2:"GreenCam",3:"GreenMobile"}
    #print(dict_bg[idxMatchedBG])
    if(idxMatchedBG==0):
        low_H = np.int16(np.clip(max_h[0] - diff_thres_black[0],0,359)).item()
        low_S = np.int16(np.clip(max_s[0] - diff_thres_black[1],0,255)).item()
        low_V = np.int16(np.clip(max_v[0] - diff_thres_black[2],0,255)).item()
        high_H = np.int16(max_h[0] + diff_thres_black[0]).item()
        high_S = np.int16(max_s[0] + diff_thres_black[1]).item()
        high_V = np.int16(max_v[0] + diff_thres_black[2]).item()
    elif(idxMatchedBG==1):
        low_H = np.int16(np.clip(max_h[0] - diff_thres_white[0],0,359)).item()
        low_S = np.int16(np.clip(max_s[0] - diff_thres_white[1],0,255)).item()
        low_V = np.int16(np.clip(max_v[0] - diff_thres_white[2],0,255)).item()
        high_H = np.int16(max_h[0] + diff_thres_white[0]).item()
        high_S = np.int16(max_s[0] + diff_thres_white[1]).item()
        high_V = np.int16(max_v[0] + diff_thres_white[2]).item()
    elif(idxMatchedBG==2):
        low_H = np.int16(np.clip(max_h[0] - diff_thres_green_cam[0],0,359)).item()
        low_S = np.int16(np.clip(max_s[0] - diff_thres_green_cam[1],0,255)).item()
        low_V = np.int16(np.clip(max_v[0] - diff_thres_green_cam[2],0,255)).item()
        high_H = np.int16(max_h[0] + diff_thres_green_cam[0]).item()
        high_S = np.int16(max_s[0] + diff_thres_green_cam[1]).item()
        high_V = np.int16(max_v[0] + diff_thres_green_cam[2]).item()
    else:
        low_H = np.int16(np.clip(max_h[0] - diff_thres_green_mobile[0],0,359)).item()
        low_S = np.int16(np.clip(max_s[0] - diff_thres_green_mobile[1],0,255)).item()
        low_V = np.int16(np.clip(max_v[0] - diff_thres_green_mobile[2],0,255)).item()
        high_H = np.int16(max_h[0] + diff_thres_green_mobile[0]).item()
        high_S = np.int16(max_s[0] + diff_thres_green_mobile[1]).item()
        high_V = np.int16(max_v[0] + diff_thres_green_mobile[2]).item()

    lowerb = (low_H, low_S, low_V)
    upperb = (high_H, high_S, high_V)
    lowerb = (low_H, low_S, low_V)
    upperb = (high_H, high_S, high_V)
    #print(f"lowerb{lowerb}")
    #print(f"upperb{upperb}")
    threshold_img = cv.inRange(img, lowerb, upperb)
    return threshold_img


def main():
    global img_path,alpha_value
    list_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    del_lists = []
    for i,fname in enumerate(list_files):
        last = len(fname) - 1
        file_ext = fname[-3:]
        if(file_ext!='JPG' and file_ext!='jpg'): # and file_ext!='JPG'
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
        cv.imwrite(img_path+"/seg/"+list_files[i]+"_segment.jpg",pointBG_imgs[i])
    
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