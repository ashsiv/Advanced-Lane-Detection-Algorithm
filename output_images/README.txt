The images in this folder denotes the outputs at various stages of the piepline.

The pipeline used for this project is described below:

A class is defined for left and right lanes to store their properties such as best fit, lane dectected?(True or False) etc. 

1. First the camera calibration matrix is computed and distortion coefficients are determined using a given a set of chessboard images.
  
2. Distortion correction matrix is applied to incoming raw images from video.

3. Individual Color thresholding ( based on S and H channels in HLS space, R channel in RGB space) is peformed on the undistorted image.
 
4. Sobel X gradient thresholding performed on the undistorted image.

5. Resultant image based on outputs in step 3 & in step 4 (series of AND or OR operations) is obtained.
 
6. Region of interest is defined.

7. Perspective transform applied on region of interest for a "birds-eye view" of the thresholded binary image. 
   
8. a. Initial lane pixels for sliding window approach are detected through histogram of lower half of image. 
 
8. b. A fit is determined using sliding window approach for rest of the image. Curvature of the lane and vehicle position with respect to center are determined. 
  
8. c. Sanity Check Protocol:
      * Check if both left and right lane is detected
      * Check if lanes are parallel (observe distance between left and right lane)
      * Check if lanes are too close or too far.
      
8. d. If no anamoly detected in the sanity check protocol for the fit estimation, store the lane properties to obtain a best fit for both left and right lanes.
      For the future incoming images, a marginal window search (see figure below) for new lanes pixels will be conducted based on the previously obtained best fits for left and right lanes. 
      Sanity check is conducted again for the obtained fit based on new lane pixels. If it evaluates False, histogram based sliding window approach will be reinvoked, else marginal search will be continued for future frames.
  
9. The detected lane boundaries are projected back onto the original image. The final output includes visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
   
=====================================================================================
The images in the 'test_images' folder denote the Outputs of the pipeline at step 8. 

NOTE: Since the input images are taken at different time instances, the fits obtained 
does not include any historical  information (of the previous frames/lane fits). 
Hence the fits shown in the images might not be the best case fits.
