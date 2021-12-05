import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir,mkdir
from os.path import isfile, join, exists
import copy

object_name_list = [
    '1WayConnectorforFoley',
    '2WayConnectorforFoley',
    '2WayFoleyCatheter',
    '3WayConnectorforFoley',
    '3Waystopcock',
    'AlcoholBottle',
    'AlcoholPad',
    'BootCover',
    'CottonBall',
    'CottonSwap',
    'Dilator',
    'DisposableInfusionSet',
    'ExtensionTube',
    'FaceShield',
    'FrontLoadSyringe',
    'GauzePad',
    'Glove',
    'GuideWire',
    'LiquidBottle',
    'Mask',
    'NasalCannula',
    'Needle',
    'NGTube',
    'OxygenMask',
    'PharmaceuticalProduct',
    'Pill',
    'PillBottle',
    'PPESuit',
    'PrefilledHumidifier',
    'PressureConnectingTube',
    'ReusableHumidifier',
    'SodiumChlorideBag',
    'SterileHumidifierAdapter',
    'SurgicalBlade',
    'SurgicalCap',
    'SurgicalSuit',
    'Syringe',
    'TrachealTube',
    'UrineBag',
    'Vaccinebottle',
    'WingedInfusionSet',
]

DiscardBufferingImageForShow = True

object_name = object_name_list[0]
dataset_path = 'D:/Dataset Medical Waste/'
dataset_crop_path = 'D:/Dataset Medical Waste(Cropped)/'
alpha_value = .7 # 0.1-1


img_path = dataset_path + object_name + '/'
img_crop_path = dataset_crop_path + object_name + '/'

mean_black = np.array([26,11,53])
mean_white = np.array([50,13,117])
mean_green_cam = np.array([110,191,122])
mean_green_mobile = np.array([119,191,101])

diff_thres_black = np.array([350,40,20]) # [70,110,40]  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_white = np.array([80,40,20])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_green_cam = np.array([40,100,20])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
diff_thres_green_mobile = np.array([45,60,40])  # [0-359,0-255,0-255] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

padW = 6 # เติมช่องว่างรอบวัตถุด้านแนวนอน
padH = 6 # เติมช่องว่างรอบวัตถุด้านแนวตั้ง
class cvRect:
    def __init__(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xmin = self.x
        self.ymin = self.y
        self.xmax = self.x + self.w
        self.ymax = self.y + self.h
    def area(self):
        return self.w * self.h
    def tl(self):
        return [self.x,self.y]
    def br(self):
        return [self.x+self.w,self.y+self.h]
    def center(self):
        return [self.x+(self.w//2),self.y+(self.h//2)]
    def get_xywh(self):
        return  [self.x,self.y,self.w,self.h]


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
    inrange_img = cv.inRange(img, lowerb, upperb)
    detected_color = idxMatchedBG #{0:"Black",1:"White",2:"GreenCam",3:"GreenMobile"}
    return inrange_img, detected_color

def locateBG(inrange_img,color):
    kernel_ELLIPSE_2x2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    _,thres_img = cv.threshold(inrange_img,127,255,cv.THRESH_BINARY_INV)
    if(color != 1): #if color is not White will not be eroded
        thres_img = cv.erode(thres_img,kernel_ELLIPSE_2x2,iterations=1)
    contours, _ = cv.findContours(thres_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    OBJS_RECT = []
    OBJS_CENTER = []
    IMG_CENTER = [inrange_img.shape[1]//2,inrange_img.shape[0]//2] # [x_center, y_center]
    OBJS_DIFF_CENTER = []
    # reject small contour (noise)
    for i,cnt in enumerate(contours): 
        tmpRect = cvRect( cv.boundingRect(cnt) ) 
        if(tmpRect.w>=50 or tmpRect.h>=50):
            OBJS_RECT.append(cvRect( cv.boundingRect(cnt) )) # [x,y,w,h]
            OBJS_CENTER.append(tmpRect.center()) # [x_obj_center, y_obj_center]
            OBJS_DIFF_CENTER.append(abs(IMG_CENTER[0]-tmpRect.center()[0])+abs(IMG_CENTER[1]-tmpRect.center()[1])) # diff = [ X_IMG_CENTER - x_obj_center, Y_IMG_CENTER - y_obj_center]
    (hImage,wImage) = inrange_img.shape[:2]
    #define defalut rect
    defalutRect = cvRect([300,300,wImage-200,hImage-200])
    # find middlest RECT
    middlest_RECT = defalutRect #in case if not found RECT -> be use default
    if(len(OBJS_RECT)==1): # if have only one RECT
        middlest_RECT = OBJS_RECT[0]
    elif(len(OBJS_RECT)>1): # find middlest RECT
        tmp_min_middle = OBJS_DIFF_CENTER[0]
        for i,val in enumerate(OBJS_DIFF_CENTER):
            #print(f"{val}",end=',')
            if(val <= tmp_min_middle):
                tmp_min_middle = val
                middlest_RECT = OBJS_RECT[i]
        #print(f"select {tmp_min_middle}")
    #cv.rectangle(thres_img,(middlest_RECT[0],middlest_RECT[1]),(middlest_RECT[0]+middlest_RECT[2],middlest_RECT[1]+middlest_RECT[3]),(255,255,255),2)
    
    # Space Padding เติมขอบข้าง
    if( (middlest_RECT.x - padW) >=0 ): # check ว่าเกินด้านซ้ายของภาพมั้ย
        middlest_RECT.x = middlest_RECT.x - padW
        middlest_RECT.w = middlest_RECT.w + padW
    if( (middlest_RECT.y - padH) >=0 ): # check ว่าเกินด้านบนของภาพมั้ย
        middlest_RECT.y = middlest_RECT.y - padH
        middlest_RECT.h = middlest_RECT.h + padH
    if( (middlest_RECT.x + middlest_RECT.w + padW) < wImage): # check ว่าเกินด้านขวาของภาพมั้ย
        middlest_RECT.w = middlest_RECT.w + padW
    if( (middlest_RECT.y + middlest_RECT.h + padH) < hImage): # check ว่าเกินด้านล่างของภาพมั้ย
        middlest_RECT.h = middlest_RECT.h + padH

    # if middlest obj has area >90% (false positive) all white image
    allArea = inrange_img.shape[1]*inrange_img.shape[0] # w * h
    objArea = middlest_RECT.w * middlest_RECT.h # w * h
    if(objArea/allArea>0.9):
        middlest_RECT = defalutRect #in case all white image -> be use default
    return thres_img,middlest_RECT
    


def ProcessInEachFolder():
    global img_path,img_crop_path,alpha_value,object_name
    divideHeight = 1
    divideWidth = 1
    list_files = []
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
        h = tmp_img.shape[0]//divideHeight
        w = tmp_img.shape[1]//divideWidth
        imgs.append(cv.resize(tmp_img,(w,h)))
    # Set low contrast
    lowct_imgs = []
    for i,img in enumerate(imgs):
        lowct_imgs.append(cv.convertScaleAbs(img,alpha=alpha_value, beta=0))
    # Convert to HSV
    HSV_imgs = []
    for i,img in enumerate(lowct_imgs):
        HSV_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL))

    if DiscardBufferingImageForShow: # <------------------------------------------------------------- Save memory
        lowct_imgs.clear()

    # Call func pointBG
    inrange_imgs = []
    detected_colors = []
    for i,img in enumerate(HSV_imgs):
        tmp_point_img,tmp_color = pointBG(img)
        inrange_imgs.append(tmp_point_img)
        detected_colors.append(tmp_color)

    if DiscardBufferingImageForShow: # <------------------------------------------------------------- Save memory
        HSV_imgs.clear()

    locateBG_imgs = []
    locateBG_xywh = []
    for i,img in enumerate(inrange_imgs):
        color = detected_colors[i]
        ret_img,ret_xywh = locateBG(img,color)
        locateBG_imgs.append(ret_img)
        locateBG_xywh.append(ret_xywh)
        xywh = locateBG_xywh[i]
        tl_point = (xywh.x,xywh.y)
        br_point = (xywh.x+xywh.w,xywh.y+xywh.h)
        #cv.rectangle(imgs[i],tl_point,br_point,(0,255,0),2) # (x,y),(x+w,y+h)
        cropped_image = imgs[i][xywh.y:xywh.y+xywh.h ,xywh.x:xywh.x+xywh.w]
        imgName,imgExtension = os.path.splitext(fname)
        cv.imwrite(img_crop_path+imgName+'.png',cropped_image)
        #cv.imwrite(img_path+"/seg/"+list_files[i]+"_segment.jpg",imgs[i])
        #cv.imwrite(img_path+"/seg/"+list_files[i]+"_segment.png",locateBG_imgs[i])

    # Display by plt
    plt_index = 1
    num_imgs = len(imgs)
    col = 4
    '''plt.rcParams["figure.figsize"] = (30,40)
    for i in range(num_imgs):
        if i==1 :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.title("Original"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.title("LowContrast"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.title("HSV"),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(inrange_imgs[i]),plt.title("pointBG"),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(locateBG_imgs[i]),plt.title("Eroded"),plt.xticks([]),plt.yticks([])
            plt_index+=1
        else :
            plt.subplot(num_imgs,col,plt_index),plt.imshow(imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(lowct_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            #plt.subplot(num_imgs,col,plt_index),plt.imshow(HSV_imgs[i]),plt.xticks([]),plt.yticks([])
            #plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(inrange_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
            plt.subplot(num_imgs,col,plt_index),plt.imshow(locateBG_imgs[i]),plt.xticks([]),plt.yticks([])
            plt_index+=1
    plt.show()'''

def main():
    global object_name,img_path,img_crop_path,dataset_path,dataset_crop_path,object_name_list
    for name in object_name_list:
        object_name = name
        img_path = dataset_path + object_name + '/'
        img_crop_path = dataset_crop_path + object_name + '/'
        if not os.path.exists(img_crop_path): # เช็คว่า path existed ?
            os.mkdir(img_crop_path)
        print(f'Processing in Folder : {object_name}')
        ProcessInEachFolder()
    print('------------------------------Finished---------------------------------')


if __name__ == "__main__":
    main()