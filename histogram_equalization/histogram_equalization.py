import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

def cal_cdf(img):
    img=img.flatten()
    pdf,_ = np.histogram(img, bins=range(257), density=False)
    cdf = np.cumsum(pdf)
    return cdf
  
def histogram_equalization(cdf, img):
    equalized_img = img.copy()

    #normalize cdf and create look up table(LUT)
    cdf_min = next(value for value in cdf if value > 0)
    cdf_max = cdf.max()
    lut = ((cdf - cdf_min)/(cdf_max-cdf_min))*255
    lut = np.round(lut)

    #convert original image into equalized image
    equalized_img = lut[img]
    return equalized_img

def divide_img(img):
    sub_img_lst=[]
    for i in range(0,193,64):
        for j in range(0,193,64):
            sub_img_lst.append(img[i:i+64,j:j+64])
    return sub_img_lst

def concate_subimg(sub_img_lst):
    horizontal_lst=[]
    for i in range(0,13,4):
        h_img=np.concatenate((sub_img_lst[i:i+4]),axis=1)
        horizontal_lst.append(h_img)
    concated_img= np.concatenate((horizontal_lst),axis=0)
    return concated_img

def global_histogram_equalization(img,img_name):
    plt.figure(f"Global-Equalization_{img_name}")
    cdf = cal_cdf(img)
    equalized_img = histogram_equalization(cdf,img)

    titles=['Original_Image','Original_Histogram','Global_Equalized_Image','Global_Equalized_Histogram']

    # show global histogram equalization result
    for i in range(4):
        plt.subplot(2,2,i+1)
        if(not (i % 2)):
            plt.imshow(img if i == 0 else equalized_img, cmap='gray')
            plt.axis('off')
        else:
            plt.hist(img.flatten() if i == 1 else equalized_img.flatten(), bins = 256, range = (0,256),color = "#87CEEB")
        
        plt.title(titles[i])
   

# divide the original 256*256 img into 16 16*16 subimgs
def local_histogram_equalization(img, img_name):
    sub_img_lst = divide_img(img)
    equalized_blocks = []

    # show original block pdf
    plt.figure(f"{img_name}_Original_Block_PDF")
    for i in range(16):
        sub_img = sub_img_lst[i].flatten()
        plt.subplot(4,4, i+1)
        plt.hist(sub_img,bins = 256, range=(0, 256), color = "#87CEEB",)
        plt.title(f"block_{i}",fontsize = 10)
        plt.subplots_adjust(hspace = 0.5)

    #show equalized block pdf
    plt.figure(f"{img_name}_Local_Equalized_Block_PDF")
    for i in range(16):
        equalized_blocks.append(histogram_equalization(cal_cdf(sub_img_lst[i]), sub_img_lst[i]))
        sub_img = equalized_blocks[i].flatten()
        plt.subplot(4,4, i+1)
        plt.hist(sub_img,bins = 256, range=(0, 256), color = "#87CEEB")
        plt.title(f"block_{i}",fontsize = 10)
        plt.subplots_adjust(hspace = 0.5)

    # local equalization result
    local_equalizd_img=concate_subimg(equalized_blocks)
    
    # show local equalization result image
    plt.figure(f"Local_Equalization_{img_name}")
    titles=['Original_Image','Local_Equalized_Image']
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(img if i == 0 else local_equalizd_img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    

if __name__ == "__main__":

    imgs=['Lena.bmp', 'Peppers.bmp']

    for i in range(2):
        img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        global_histogram_equalization(img, imgs[i])
        local_histogram_equalization(img, imgs[i])
        
    plt.show()
    

   

