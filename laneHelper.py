import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def find_lane_pixels_byWin(binary_warped):
    """find all possible left/right lane pixels by sliding window"""
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
   
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
   
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_xleft_low  = leftx_current - margin  # left side of window for left lane
        win_xleft_high = leftx_current + margin  # right side of window for left lane
        win_xright_low = rightx_current - margin  # left side of window for right lane
        win_xright_high = rightx_current + margin  # right side of window for right lane
        # Draw the windows on the visualization image  --- can be delted if no need
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###  数组逻辑运算！！！
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###  
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices (previously was a list of lists(arrays) of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img


def find_lane_pixels_byPoly(binary_warped, fits):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # default margin 100, can tune it !
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # caculate fit x values fits[0] = left_fit , fits[1] = right_fit
    left_fitx = fits[0][0]*nonzeroy**2 + fits[0][1]*nonzeroy + fits[0][2]
    right_fitx = fits[1][0]*nonzeroy**2 + fits[1][1]*nonzeroy + fits[1][2]
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = (nonzerox > (left_fitx - margin)) & (nonzerox < (left_fitx + margin))
    right_lane_inds = (nonzerox > (right_fitx - margin)) & (nonzerox < (right_fitx + margin))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty    


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
   
    ### Calc both polynomials using ploty, left_fit and right_fit 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    fits = [left_fit, right_fit]
    fitxs = [left_fitx, right_fitx]
    
    return fits, fitxs, ploty


def draw_lines(warped, fitxs, ploty):    
    left_fitx, right_fitx = fitxs[0], fitxs[1]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly() ***
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
      
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp


def calc_curve(y, A, B):
    return ((1 + (2*A*y + B)**2)**(3/2))/np.absolute(2*A)
    

def measure_curvature_real(fitxs, ploty, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    left_fitx, right_fitx = fitxs[0], fitxs[1]
     
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
            
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)* ym_per_pix
    
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = calc_curve(y_eval, left_fit_cr[0], left_fit_cr[1])    ## Implement the calculation of the left line here
    right_curverad = calc_curve(y_eval, right_fit_cr[0], right_fit_cr[1])  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def get_info(left_bottom_x, right_bottom_x, imshape, xm_per_pix, avg_cr):
    lane_center = (left_bottom_x + right_bottom_x)/2
    offset = (imshape[1]/2 - lane_center)* xm_per_pix   
    if offset < 0:
        pos = "left"
    elif offset > 0:
        pos = "right"
            
    offset = "%.2f"%np.absolute(offset)
    avg_cr = "%d"%avg_cr

    show_info = ["Radius of Curvature = " + avg_cr +"(m)" 
                  , "Vehicle is "+ offset + "m " + pos + " of center"]
    
    return show_info



#########  no in use ############
def calc_best(ploty, fits): 

    #print('in calc_best')
    #print('left_fit', fits[0])
    #print('right_fit', fits[1])
    #print('ploty.shape', ploty.shape)
    
    # caculate fit x values fits[0] = left_fit , fits[1] = right_fit
    left_fitx = fits[0][0]*ploty**2 + fits[0][1]*ploty + fits[0][2]
    right_fitx = fits[1][0]*ploty**2 + fits[1][1]*ploty + fits[1][2]

    fitxs = [left_fitx, right_fitx]
    
    return fitxs

def sanity_check(fitxs):
    check_result = True
    for i in range(len(fitxs[0])):
        diff = fitxs[0][i] - fitxs[1][i] # pixel distance between left and right
        if  diff < 620 or diff > 750:  # too wide or too narrow or even cross
            check_result = False
            break    
    return check_result



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
