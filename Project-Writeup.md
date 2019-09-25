# Project 2 Advance Lane Finding

### Writeup/README


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0.0]: ./output_images/Pipeline/0.0-corners.png "Corners"
[image0.1]: ./output_images/Pipeline/0.1-cali.jpg "Calibration"
[image0.2]: ./output_images/Pipeline/0.2-undist.jpg "Undist chess"

[image1.0]: ./test_images/test6.jpg "Original "
[image1.1]: ./output_images/Pipeline/1-undist.jpg "Undistorted"

[image2.1]: ./output_images/Pipeline/2.1-X-grad1.jpg "Binary Example"
[image2.2]: ./output_images/Pipeline/2.2-S-Channel1.jpg "Binary Example"
[image2.3]: ./output_images/Pipeline/2.3-R-Channel1.jpg "Binary Example"
[image2.4]: ./output_images/Pipeline/2.4-Combined-binary1.jpg "Combo Bin"

[image3.0]: ./output_images/Pipeline/3.0-Warp-SrcDst.png "Warp with src/dst points"

[image4.0]: ./output_images/Pipeline/4.0-hist.png "histogram"
[image4.1]: ./output_images/Pipeline/4.1-test6-pixels.jpg "lane pixels"
[image4.2]: ./output_images/Pipeline/4.2-test6-lanes.png "lane lines"

[image5.1]: ./output_images/Pipeline/5.1-warped-lanes.png "warp lanes"
[image5.2]: ./output_images/Pipeline/5.2-warped-back.png "warp back"

[image6.1]: ./output_images/Pipeline/6.1-Final-test6_out.jpg "final"
[image6.2]: ./output_images/Pipeline/6.2-Final-test-all.png "final all"

[video1]: ./Adv_out_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the **2nd code cell** of IPython notebook located in `./Proj2-Adv-LaneLines.ipynb`
  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Find corners on chessboard  

![alt text][image0.0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


Original Distorted chessboard
![alt text][image0.1]
After distortion correction
![alt text][image0.2]

## Pipeline (single images)

The overall pipeline is implemented in the **4th code cell** of the IPython notebook `./Proj2-Adv-LaneLines.ipynb` , there is a skeleton for all steps in function of `process_image(image)` 

Note: in the **3rd code cell** in the IPython notebook, I defined all varialbes/data that will be used for all the following processes

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image1.0]

The code for distortion-correction is in imageHelper.py line #5 to line #8,it uses the **camera matrix** and **distortion coefficients** obtained in the step of camera calibration
``` python
`def cal_undistort(img, mtx, dist):  
    """undistort image"""
    undist= cv2.undistort(img, mtx, dist, None, mtx)   
    return undist
``` 

Here is result image After distortion correction 
![alt text][image1.1]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of **color and gradient thresholds** to generate a binary image
 
Here are 2 functions I wrote,  one for gradient thresholding, another for color channel threshholding, they are:   
 
`grad_thresh(img_gray, orient, thresh, sobel_kernel=3)`  
`color_thresh(img, r_thresh, s_thresh)`  
(thresholding steps located at lines #12 through #36 in 
 imageHelper.py 
  
Also I wrote a  function to combine the above 2 thresholding functions, which is in imageHelper.py, at line# 63 through line #75, it mainly does
``` python
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
```

### **The thresholding steps are as following**
First,  I implemente **Gradient thresholding** along X direction(i.e. horizontally) with threshold values (30, 100).  I once tried (20, 100), but found that if using (20, 100),  some road lanes can come up with a weird warped image,   while a lower bound of up to 30 can remove more noises without losing useful lane pixels

Gradient thresholding uses Sobel operation to detect the value changes
of the pixels so that it can detect edges in the image   
`sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)` 

Here is the result with **Gradient thresholding only** (x direction)
![alt text][image2.1]

Then I use color thresholding by seperating R channel from a RGB image and converting a RGB image to HLS color space , so that I can get S channel out.
``` python
""" img: RGB color image"""
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # 2) Apply threshold to the R,S channel respectively
    R = img[:,:,0]
    S = hls[:,:,2]
``` 

### **The following show the results of color thresholding**

I use **S Channel** threshoding with (170,255), which can pick up lane lines very well and result looks clean, but may miss some pixels that shows in **X gradient** thresholded image as above

![alt text][image2.2]

I use **R Channel** thresholding with (180,255)
![alt text][image2.3]

Here is the final **Combined binary** image , which can show lane lanes fairly fine without many noises, I later also use region of interest mask to leave only targeted area
![alt text][image2.4]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines #77 through #83 in the file `imageHelper.py`  .  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 300, 0        | 
| 700, 460      | 950, 0      |
| 1096, 720     | 950, 720      |
| 200, 720      | 300, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3.0]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial using  **sliding window** method. It is implemented in `laneHelper.py` from line #6 through line #94 , a function called `find_lane_pixels_byWin(binary_warped)`

First I obtained start points for the left and right lanes finding , by taking a histogram along all the columns in the lower half of the image like this:

``` python
# Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

# Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
``` 

![alt text][image4.0]

Starting from the bottom, move the windows up using a loop and find all nonezero points/pixels in the binary image that fall into the windows which are most likely on the lanes I am trying to find

``` python
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

...
### Identify the nonzero pixels in x and y within the window ###  数组逻辑运算！！！
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
...
```

**Sliding windows** look like the following
![alt text][image4.1]

After I got all possible points(pixels) for the left and right lanes respectively, I fit a second order polynomial to each of the 2 set of points, a function called `fit_poly(img_shape, leftx, lefty, rightx, righty)`is implemented in `laneHelper.py` line #127 to line #142, mainly using np.polyfit() as the following:
``` python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
``` 

The **2 fited lanes** show as Yellow lines as the following  
![alt text][image4.2]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did **Curvature Calculation** in lines #162 through #183 in my code in `laneHelper.py` ,where 2 functions are:

`calc_curve(y, A, B)`  and   
`measure_curvature_real(fitxs, ploty, ym_per_pix, xm_per_pix)`  

I use `ym_per_pix`   and `xm_per_pix`   to map pixel world to real world (by estimation of distances represented in the image), The mapping is defined in the 3rd code cell of `./Proj2-Adv-LaneLines.ipynb`
``` python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension   30/720
xm_per_pix = 4/700 # meters per pixel in x dimension    3.7/700
``` 

The fomula for calculating the radius of curvature at any point x of the function x = f(y) is implemented as the following, where A, B are coefficients of 2nd order polynomial curve:
``` python
def calc_curve(y, A, B):
    return ((1 + (2*A*y + B)**2)**(3/2))/np.absolute(2*A)
``` 

I then calculated **Vehicle Position** in lines #185 through #199 in my code in `laneHelper.py` , the function is `get_info(left_bottom_x, right_bottom_x, imshape, xm_per_pix, avg_cr)`  . Assume the camera is mounted in the center of the vehicle ,  the center of the image is just the center of the vehicle, while the center in-between 2 lanes can be calculated by averaging the bottom x coordinates of the 2 lanes identified.
then the vehicle offset to the road center can be calculated:
``` python
lane_center = (left_bottom_x + right_bottom_x)/2
offset = (imshape[1]/2 - lane_center)* xm_per_pix 
``` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #41 through #56 in notebook 
`./Proj2-Adv-LaneLines.ipynb`,  there I called functions `draw_line()`, `warper()` and `cv2.addWeighted()`


I implemented the function `draw_lines(warped, fitxs, ploty)`in lines #144 through #159 in my code in `laneHelper.py` in . It takes a warped image as input ,  with y coordinates and fitted x coordinates of both lanes, draw a green polygon representing the lanes

 Here is an example of my result on a test image:

![alt text][image5.1]

Warp back to original image space using inverse perspective matrix:  
`newwarp = imgHelper.warper(color_warp, dst, src)`
![alt text][image5.2]

The final output image with vehicle info
![alt text][image6.1]

The final output of all test images   
![alt text][image6.2]
---

### Pipeline (video)

#### Final video output .

Here's a [link to my video result](Adv_out_video.mp4)

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### 1. Some issues that once got me stuck:
1. For the perspective transform, it took me a while to search on web and consult others to try to figure out the source and destination points, at the begining I thought there should be an automatic way, while finally it turned out that I have to hard code , estimate and adjust mannualy

2. The radius of curvature looks much larger for straight lanes than for curved lanes, at the begining I thought something was wrong , later I think that does make sense since straight lanes can't say 'curvature'

3. Although I implemented a function called `find_lane_pixels_byPoly(binary_warped, fits)` at lines #96 through # 122 in `laneHelper.py`, it is targeted to seach around poly with prior detected lanes,  but when I used it , the results don't look good,  so I had to turn back to sliding window since overall that works fine

3. In a couple of frames the lanes seem still not perfect, if more fine tuning for the thresholding or implementing a smoothing method might be better


#### 2. What I tried for searching by poly and smoothing 
The following code are what I once tried for sanity check and smoothing, but didn't get them work well, so I didn't use them for the submition, would appreciate if you could help give some hints. thanks~

1. In notebook file `Proj2-Adv-LaneLines.ipynb`

First to declared global variable in 2nd code cell ( I am not very sure how to use the Line() class, so tried to use global variables):
``` python
# for tracking detected lines
keep_frame = 5
detected = False

left_recent_fits  = deque(maxlen=keep_frame) # keep track of left recent fits
right_recent_fits = deque(maxlen=keep_frame) # keep track of right recent fits
``` 

Implemented following in  `process_image()` at STEP 4 (all others are the same as in submitted jupyter notebook file)
``` python
    """ STEP 4 Detect lane pixels and fit to find the lane boundary """    
    
    if len(left_recent_fits) > 0:
        left_best_fit  = np.array([fit for fit in left_recent_fits]).mean(axis=0)
        right_best_fit = np.array([fit for fit in right_recent_fits]).mean(axis=0)
        best_fits = [left_best_fit, right_best_fit]
        #print('best_fits', best_fits)
    
    if detected == False:
        leftx, lefty, rightx, righty = laneHelper.find_lane_pixels_byWin(binary_warped)   
    else: 
        #### Find our lane pixels by average fits for last 5 frames  ####
        leftx, lefty, rightx, righty = laneHelper.find_lane_pixels_byPoly(binary_warped, best_fits)
    
    # fit polynomial values to find lane lines
    fits, fitxs, ploty = laneHelper.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
  
    if laneHelper.sanity_check(fitxs) == True:
        detected = True
        #print('detected True')
    else:
        if (len(left_recent_fits) > 0):
            fitxs = laneHelper.calc_best(ploty, best_fits)
            fits = best_fits
        #print('detected False')
        detected = False 
        
    left_recent_fits.append(fits[0])  # keep track of recent fits
    right_recent_fits.append(fits[1])
``` 

2. Implemented following in `laneHelper.py` line #204 to line #226
``` python
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
```  
