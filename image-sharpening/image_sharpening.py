import cv2
import numpy as np
import matplotlib.pyplot as plt

laplacian_mask  = [np.array([[1, -2, 1],
                            [-2, 4, -2],
                            [1, -2, 1]]),
                    np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]]),
                    np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
]

def zero_padding(org_img):
    height, width = org_img.shape
    padded_img = org_img.copy()
    
    zero_row  = np.zeros((1, width))
    zero_col = np.zeros(((height + 1*2), 1))
    
    padded_img = np.vstack([zero_row, padded_img, zero_row])
    padded_img = np.hstack([zero_col, padded_img, zero_col])

    return padded_img

def convolution(padded_img, kernel_type, A = 1):

    height, width = padded_img.shape
    border_img = np.zeros((height-2, width-2), dtype=np.float16)
    x = 0
    y = 0

    # high-boost, A-1(default): laplacian-filter
    laplacian_mask[kernel_type] = laplacian_mask[2].astype(np.float16)
    laplacian_mask[kernel_type][1][1] += (A-1)

    while x + 3 <= height:
        while y + 3 <= width:
            border_img[x][y] = np.sum(padded_img[x:x+3, y:y+3]* laplacian_mask[kernel_type])
            y += 1
        x += 1
        y = 0
    
    
    laplacian_mask[kernel_type][1][1] -= (A-1)

    # prevent gray value overfolow
    border_img = np.clip(border_img, 0, 255)
    border_img = np.round(border_img.astype(np.uint8))
    return border_img

def laplacian_sharpening(org_imgs, kernel_type):
    
    laplacian_imgs = [convolution(zero_padding(org_imgs[0]), kernel_type), convolution(zero_padding(org_imgs[1]), kernel_type)]
    laplacian_sharpend_imgs = [cv2.add(org_imgs[0], laplacian_imgs[0]), cv2.add(org_imgs[1], laplacian_imgs[1])]
    
    return [org_imgs, laplacian_imgs, laplacian_sharpend_imgs]
  

def high_boost_sharpening(org_imgs, A, kernel_type):

    boosted_imgs = [convolution(((zero_padding(org_imgs[0])).astype(np.float16)), kernel_type, A), convolution(((zero_padding(org_imgs[1])).astype(np.float16)), kernel_type, A)]
    hb_sharpened_imgs = [cv2.add(org_imgs[0], boosted_imgs[0]), cv2.add(org_imgs[1], boosted_imgs[1])]
    
    return hb_sharpened_imgs
     
def show_lp_result(org_imgs, laplacian_imgs, sharpened_imgs):
    
        plt.figure("Laplacian-enhancement")
        plt.subplot(2, 3, 1)
        plt.title ("org_skeleton")
        plt.imshow(org_imgs[0], cmap = "gray")

        plt.subplot(2, 3, 2)
        plt.title ("laplacian_skeleton")
        plt.imshow(laplacian_imgs[0], cmap = "gray")

        plt.subplot(2, 3, 3)
        plt.title ("Sharpened_skeleton")
        plt.imshow(sharpened_imgs[0], cmap = "gray")

        plt.subplot(2, 3, 4)
        plt.title ("org_moon")
        plt.imshow(org_imgs[1], cmap = "gray")

        plt.subplot(2, 3, 5)
        plt.title ("laplacian_moon")
        plt.imshow(laplacian_imgs[1], cmap = "gray")

        plt.subplot(2, 3, 6)
        plt.title ("Sharpened_moon")
        plt.imshow(sharpened_imgs[1], cmap = "gray")
        
def show_hb_result(hb_imgs):
    plt.figure("high-boost_sharpening")
    
    plt.subplot(2, 3, 1)
    plt.title("skeketon A=1.0")
    plt.imshow(hb_imgs[0][0], cmap="gray")

    plt.subplot(2, 3, 2)
    plt.title("skeketon A=1.5")
    plt.imshow(hb_imgs[1][0], cmap = "gray")

    plt.subplot(2, 3, 3)
    plt.title("skeketon A=2")
    plt.imshow(hb_imgs[2][0], cmap = "gray")

    plt.subplot(2, 3, 4)
    plt.title("blurry_moon A=1.0")
    plt.imshow(hb_imgs[0][1], cmap = "gray")

    plt.subplot(2, 3, 5)
    plt.title("blurry_moon A=1.5")
    plt.imshow(hb_imgs[1][1], cmap = "gray")

    plt.subplot(2, 3, 6)
    plt.title("blurry_moon A=2")
    plt.imshow(hb_imgs[2][1], cmap = "gray")

if __name__ == "__main__":

    org_imgs = [cv2.imread("skeleton_orig.bmp", cv2.IMREAD_GRAYSCALE), cv2.imread("blurry_moon.tif", cv2.IMREAD_GRAYSCALE)]
    lp_pack = laplacian_sharpening(org_imgs, 2)
    
    hb_imgs = []
    A = [1, 1.5, 2]
    for a in A:
        hb_imgs.append(high_boost_sharpening(org_imgs, a, 2))
    
    show_lp_result(lp_pack[0], lp_pack[1], lp_pack[2])
    show_hb_result(hb_imgs)

    plt.show()

    
    
