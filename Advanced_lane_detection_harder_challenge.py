import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import time
center=0
radius=0
DEBUG=0;
left_fit = None
right_fit = None
current_l_fit=np.array([0,0,0]);
current_r_fit=np.array([0,0,0]);
itr=0
mtx=None
dist=None
flag=0
sanity_check=False
src=np.float32([[400,530],[250,700],[1280,700],[1000,530]])
dst=np.float32([[0,0],[0,700],[1280,700],[1280,0]])
##src=np.float32([[580,450],[250,700],[1150,700],[720,450]])
##dst=np.float32([[250,0],[250,700],[1150,700],[1150,0]])
M=cv2.getPerspectiveTransform(src,dst)
Minv=cv2.getPerspectiveTransform(dst,src)
# Class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        self.recent_fitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = []     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    def printall(self):
        print();
        print(self.detected,self.best_fit,self.current_fit,self.radius_of_curvature)
        print();
        
def calibrate():
    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    # Make a list of calibration images
    images = glob.glob('.\camera_cal\calibration*.jpg')

    # Array to store object points and image points from all calibration images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Prepare object points
    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2) # (x,y) coordinates

    for fname in images:
        
        if(fname=='.\camera_cal\calibration1.jpg'):
            continue
        # Read in a calibration image
        img = cv2.imread(fname)
     
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
          
        # If found, draw corners, add object and image points
        if (ret == True and np.prod(corners.shape) == 108):
           
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    return [objpoints, imgpoints]

def undistort(img,objpoints, imgpoints):

    global mtx,dist,flag

    if (flag==0):    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    if(DEBUG):
        img1=img
        cv2.polylines(img1,[np.int32(src)],True,(0,255,0), 3)
        ax[0,0].imshow(img1)
        ax[0,0].set_title('Original Image')
        ax[0,1].imshow(undistorted)
        ax[0,1].set_title('Undistorted Image')
       
    return undistorted


def Threshold(und_img):
    global ax
    
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(und_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    r_channel = und_img[:,:,0]

    # Grayscale image
    gray = cv2.cvtColor(und_img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min =80
    thresh_max = 140
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # R channel
    r_thresh_min = 170
    r_thresh_max = 255
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 1

    # H channel
    h_thresh_min = 10
    h_thresh_max = 100
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
    

    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    r_nonzero = r_binary.nonzero()
    #print(len(r_nonzero[0]))
    h_nonzero = h_binary.nonzero()
    #print(len(h_nonzero[0]))
    s_nonzero = s_binary.nonzero()
    #print(len(s_nonzero[0]))
    sx_nonzero = sxbinary.nonzero()
    #print(len(sx_nonzero[0]))
    # Combine all binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[((r_binary ==1))| (sxbinary == 1)] = 1
    combined_binary[((s_binary == 1) & (r_binary ==1) & (h_binary ==1)) | (sxbinary == 1)] = 1
    if(DEBUG):
        
        # Plotting thresholded images
        ax[0,2].set_title('S channel')
        ax[0,2].imshow(s_binary, cmap='gray')
        ax[0,3].set_title('H channel')
        ax[0,3].imshow(h_binary, cmap='gray')
        ax[0,4].set_title('R channel')
        ax[0,4].imshow(r_binary, cmap='gray')
        ax[1,0].set_title('SX channel')
        ax[1,0].imshow(sxbinary, cmap='gray')
    r_nonzero=len(r_nonzero[0])
    h_nonzero=len(h_nonzero[0])
    s_nonzero=len(s_nonzero[0])
    sx_nonzero=len(sx_nonzero[0]) 
    return combined_binary,r_nonzero,h_nonzero,s_nonzero,sx_nonzero

def perspective(ths_img):

    global ax,src,dst,M,Minv
    
   
    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(ths_img, M, ths_img.shape[1::-1],flags=cv2.INTER_LINEAR)

    if(DEBUG):
       
        ax[1,1].imshow(ths_img, cmap='gray')
        ax[1,1].set_title('Threshold Image')
        ax[1,2].imshow(warped, cmap='gray')
        ax[1,2].set_title('Perspective Image')
       
        
    return warped, M

# Define a new search region based on previous image
def remapsrcdst(left_fitx,ploty,right_fitx):
    global src,dst,M,Minv
    x1=[]
    y1=[]
    a1=[]
    b1=[]
    for i in [0,-1]:
        x=left_fitx[i]-10
        y=ploty[i]
        a1.append(x)
        b1.append(y)
        x1.append(round(((Minv[0][0]*x)+(Minv[0][1]*y)+(Minv[0][2]))/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]),0))
        y1.append(round(((Minv[1][0]*x)+(Minv[1][1]*y)+(Minv[1][2]))/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]),0))
        
    for i in [-1,0]:
        x=right_fitx[i]+10
        y=ploty[i]
        a1.append(x)
        b1.append(y)
        x1.append(round(((Minv[0][0]*x)+(Minv[0][1]*y)+(Minv[0][2]))/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]),0))
        y1.append(round(((Minv[1][0]*x)+(Minv[1][1]*y)+(Minv[1][2]))/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]),0))
        
   
    dst=np.float32(np.transpose(np.vstack((a1,b1))))
    src=np.float32(np.transpose(np.vstack((x1,y1))))
    dst[dst<0.0]=0.0
    src[src<0.0]=0.0
    
def projection(image,warped,left_fitx,right_fitx,ploty):

    
    
    global ax,src,dst,M,Minv
    Minv=cv2.getPerspectiveTransform(dst,src)
    M=cv2.getPerspectiveTransform(src,dst)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    #remapsrcdst(left_fitx,ploty,right_fitx)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    if(DEBUG):
        
        ax[1,3].imshow(color_warp, cmap='gray')
        ax[1,3].set_title('Warped Image')
        ax[1,4].imshow(result, cmap='gray')
        ax[1,4].set_title('Projected Image')
        plt.show()
    return result

def find_lane_pixels(binary_warped):
    global left_fit, right_fit
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 20
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

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four boundaries of the window ###
        win_xleft_low  =leftx_current -margin
        win_xleft_high =leftx_current +margin 
        win_xright_low =rightx_current -margin  
        win_xright_high=rightx_current +margin  
       
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy <win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy <win_y_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        
        
        if (len(good_left_inds)>minpix):
            
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[win_y_low:win_y_high,:], axis=0)
            
            midpoint = np.int(histogram.shape[0]//2)
            leftx_current = np.argmax(histogram[:midpoint])
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low  =leftx_current -margin  # Update this
            win_xleft_high =leftx_current +margin  # Update this
            
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            #(win_xleft_high,win_y_high),(0,255,0), 2) 
            
            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy <win_y_high)).nonzero()[0]
            if (len(good_left_inds)>minpix):
                left_lane_inds.append(good_left_inds)
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
            
        if (len(good_right_inds)>minpix):
           
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[win_y_low:win_y_high,:], axis=0)
            
            midpoint = np.int(histogram.shape[0]//2)
            
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint
            ### TO-DO: Find the four below boundaries of the window ###
            
            win_xright_low =rightx_current -margin  # Update this
            win_xright_high=rightx_current +margin  # Update this
            # Draw the windows on the visualization image
             
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),
            #(win_xright_high,win_y_high),(0,255,0), 2) 
            
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy <win_y_high)).nonzero()[0]
            
            if (len(good_right_inds)>minpix):
                right_lane_inds.append(good_right_inds)
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
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


def fit_polynomial(leftx, lefty, rightx, righty, out_img):
    global left_fit, right_fit,sanity_check,flag,radius, center
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    ###Fit a second order polynomial to each using `np.polyfit` ###
    if (leftx !=[] and lefty !=[] and rightx !=[] and righty !=[]):
            
        left_fit = np.polyfit(lefty,leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
        # Generate x and y values for plotting
        
           
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## curvature and center calculation
        
        ym_per_pix = 30/out_img.shape[0] # meters per pixel in y dimension
        xm_per_pix = 3.7/np.absolute(np.mean(right_fitx)-np.mean(left_fitx)) # meters per pixel in x dimension
        l_fit = np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
        r_fit = np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)
        lane_center =np.mean((right_fitx + left_fitx)/2.0);
        image_center = out_img.shape[1]/2.0;
        center = (image_center-lane_center)*xm_per_pix;
        y_eval = np.max(ploty)*ym_per_pix
        y=y_eval;

        
        ##### Implement the calculation of R_curve (radius of curvature) #####
        left_curverad =  np.power(1+np.power((2*l_fit[0]*y)+l_fit[1],2),3/2)/(np.absolute(2*l_fit[0])) ## Implement the calculation of the left line here
        right_curverad =  np.power(1+np.power((2*r_fit[0]*y)+r_fit[1],2),3/2)/(np.absolute(2*r_fit[0])) ## Implement the calculation of the left line here
        radius = (left_curverad+right_curverad)/2.0;
        center=round(center,2);
        radius = round(radius,0);

        leftlane.radius_of_curvature = left_curverad
        rightlane.radius_of_curvature = right_curverad
        leftlane.line_base_pos = np.absolute(center -np.mean(left_fitx*xm_per_pix))
        rightlane.line_base_pos = np.absolute(center -np.mean(right_fitx*xm_per_pix))
        leftlane.current_fit=left_fit
        rightlane.current_fit=right_fit
        leftlane.allx=leftx
        leftlane.ally=lefty
        rightlane.allx=rightx
        rightlane.ally=righty
        
        if(np.any(left_fit)==None):
            leftlane.detected= False
        else:
            leftlane.detected=True
        if(np.any(right_fit)==None):
            rightlane.detected= False
        else:
            rightlane.detected=True

        #Sanity_check
       
        if(leftlane.detected ==False or rightlane.detected==False):
            sanity_check=False
            print("No lane detected");
        elif(np.absolute(np.min(np.absolute(left_fitx - right_fitx)) - np.max(np.absolute(left_fitx - right_fitx)))> 700):
            sanity_check=False
            print("Lanes not parallel");print(np.absolute(np.min(np.absolute(left_fitx - right_fitx)) - np.max(np.absolute(left_fitx - right_fitx))));
        elif(np.absolute(np.mean(left_fitx) - np.mean(right_fitx))< 300 or np.absolute(np.mean(left_fitx) - np.mean(right_fitx))> 900):
            sanity_check=False
            print("Lanes too close");print(np.absolute(np.mean(left_fitx) - np.mean(right_fitx)));
        else:
            sanity_check=True
            leftlane.recent_xfitted.append(left_fitx)
            rightlane.recent_xfitted.append(right_fitx)
            leftlane.recent_fitted.append(left_fit)
            rightlane.recent_fitted.append(right_fit)

            if(leftlane.recent_xfitted != [] or leftlane.recent_fitted != [] or rightlane.recent_xfitted != [] or rightlane.recent_fitted != []):
                leftlane.bestx=np.mean(leftlane.recent_xfitted,0)
                rightlane.bestx=np.mean(rightlane.recent_xfitted,0)
                leftlane.best_fit=np.mean(leftlane.recent_fitted,0)
                rightlane.best_fit=np.mean(rightlane.recent_fitted,0)
            if(DEBUG):
                leftlane.bestx=left_fitx
                rightlane.bestx=right_fitx
                leftlane.best_fit=left_fit
                rightlane.best_fit=right_fit
            leftlane.diffs =np.absolute(leftlane.best_fit-leftlane.current_fit)
            rightlane.diffs =np.absolute(rightlane.best_fit-rightlane.current_fit)

        if(flag==0 & DEBUG==0):
            leftlane.recent_xfitted.append(left_fitx)
            rightlane.recent_xfitted.append(right_fitx)
            leftlane.recent_fitted.append(left_fit)
            rightlane.recent_fitted.append(right_fit)

            leftlane.bestx=np.mean(leftlane.recent_xfitted,0)
            rightlane.bestx=np.mean(rightlane.recent_xfitted,0)
            leftlane.best_fit=np.mean(leftlane.recent_fitted,0)
            rightlane.best_fit=np.mean(rightlane.recent_fitted,0)

            leftlane.diffs =np.absolute(leftlane.best_fit-leftlane.current_fit)
            rightlane.diffs =np.absolute(rightlane.best_fit-rightlane.current_fit)
            flag=1;
##        leftlane.bestx=left_fitx
##        rightlane.bestx=right_fitx
##        leftlane.best_fit=left_fit
##        rightlane.best_fit=right_fit
    return out_img,leftlane.bestx, rightlane.bestx, ploty, leftlane.best_fit,rightlane.best_fit, radius, center

def search_around_poly(binary_warped,left_fit,right_fit):
    
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    
    margin = 20    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
   
    win_xleft_low =   (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2])- margin
    win_xleft_high =  (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]) + margin
    win_xright_low = (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2])- margin
    win_xright_high =  (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2])+ margin
    win_y_low=0
    win_y_high=2000
    
    left_lane_inds= []
    right_lane_inds = []
    good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <=win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy <=win_y_high)).nonzero()[0]
    good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <=win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy <=win_y_high)).nonzero()[0]
         
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    [out_img,left_fitx, right_fitx, ploty,left_fit,right_fit,radius, center]=fit_polynomial(leftx, lefty, rightx, righty, binary_warped)
    
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
   # print(right_line_pts)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
       
    return [result,left_fitx,right_fitx,ploty,radius, center]

def measure_curvature_real(ploty,left_fit_cr,right_fit_cr,left_fitx,right_fitx):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
       
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y= np.max(ploty)*ym_per_pix
    
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad =  np.power(1+np.power((2*left_fit_cr[0]*y)+left_fit_cr[1],2),3/2)/(np.absolute(2*left_fit_cr[0])) ## Implement the calculation of the left line here
    right_curverad =  np.power(1+np.power((2*right_fit_cr[0]*y)+right_fit_cr[1],2),3/2)/(np.absolute(2*right_fit_cr[0])) ## Implement the calculation of the left line here

    
    center = (640-((left_fitx+right_fitx)/2.0))*xm_per_pix
    return left_curverad, right_curverad,center

def process_image(img):
 
    global src,dst,M,Minv,objpoints, imgpoints, left_fit, right_fit, current_r_fit, current_l_fit,itr,ax,sanity_check,flag
    
    if(DEBUG):
        f, ax = plt.subplots(2, 5)
        f.tight_layout()
    
    # STEP 1: Undistort the image
    undistorted_image=undistort(img, objpoints, imgpoints)
    
    # STEP 2: Image Thresholding
    [ths_img,r_nonzero,h_nonzero,s_nonzero,sx_nonzero] = Threshold(undistorted_image)
    #print("i")
    # STEP 3: Perspective Transform
    [binary_warped,M]= perspective(ths_img)
   
  
    # STEP 4: Lane fit
    
    if (True):
        
        # Find lane pixels
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

        # Fit polynomial
        [out_img,left_fitx, right_fitx, ploty,leftlane.best_fit,rightlane.best_fit,radius, center]=fit_polynomial(leftx, lefty, rightx, righty, out_img)

    else:

        print("Searching for fit");
        #search around already fitted polynomial
        [out_img,leftlane.bestx, rightlane.bestx,ploty,radius, center] = search_around_poly(binary_warped,leftlane.best_fit,rightlane.best_fit)
        
    # STEP 5: Project lane back to image
    out_img = projection(img,binary_warped,leftlane.bestx, rightlane.bestx,ploty)
    
    # Print curvature and center details on the image
    if ((np.all(left_fit != None)) and (np.all(right_fit != None) )):
        
        cv2.putText(out_img, "Radius of Curvature=" + str(radius) + " (m)", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if (center<0):
            cv2.putText(out_img, "Vehicle is " + str(np.absolute(center)) + " (m) left of center", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(out_img, "Vehicle is " + str(np.absolute(center)) + " (m) right of center", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
            itr=itr+1
    if(itr==2):
        leftlane.recent_xfitted = []
        rightlane.recent_xfitted = []
        leftlane.recent_fitted = []
        rightlane.recent_fitted = []
        itr=0; 
    return out_img

#######################
#      MAIN           #
#######################

# STEP 1: Calibrate camera
[objpoints,imgpoints] = calibrate();
leftlane=Line()
rightlane=Line()

### STEP 2: Process image/video

if(DEBUG):
    for file in glob.glob('test_images/*.jpg'):
        print (file)
        img = mpimg.imread(file)
        img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
        out_img = process_image(img);
        cv2.imwrite('./output_images\\'+file, out_img)
        
else:       
    white_output = 'output_harder_challenge1_video.mp4'
    clip1 = VideoFileClip("harder_challenge_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


