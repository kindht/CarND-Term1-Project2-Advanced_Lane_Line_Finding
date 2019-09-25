import numpy as np
import cv2


def cal_undistort(img, mtx, dist):
    """undistort image"""
    undist= cv2.undistort(img, mtx, dist, None, mtx)   
    return undist

def grad_thresh(img_gray, orient, thresh, sobel_kernel=3):
    # Threshold x or y gradient
    if orient == 'x': 
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & ( scaled_sobel <= thresh[1])] = 1

    return grad_binary

def color_thresh(img, r_thresh, s_thresh):
    """ img: RGB color image"""
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # 2) Apply threshold to the R,S channel respectively
    R = img[:,:,0]
    S = hls[:,:,2]
    
    # 3) Threshold color channel
    color_binary = np.zeros_like(S)
    color_binary[(R >= r_thresh[0]) & ( R <= r_thresh[1]) & (S >= s_thresh[0]) & ( S <= s_thresh[1])] = 1        
  
    # 4) Return a binary image of threshold result
    return color_binary

def region_of_interest(img, vertices):
    """
    Applies an image mask        
    vertices: a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def combined_binary(img_orig, sobel_thresh, r_thresh, s_thresh):
    ##  img_orig: RGB color image
    img_copy = np.copy(img_orig)
   
    gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    grad_binary  = grad_thresh(gray, orient='x', thresh=sobel_thresh) 
    color_binary = color_thresh(img_copy, r_thresh=r_thresh, s_thresh=s_thresh)
      
    combined = np.zeros_like(color_binary)
    combined[(grad_binary ==1) | (color_binary ==1)] = 1 
        
    combined *= 255 # for binary image , scale to 0~255
    return combined

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped