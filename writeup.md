**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistorted_cb.jpg "Orignal chessboard and undistorted"
[image2]: ./output_images/undistorted/straight_lines1.jpg "Road undistorted"
[image3]: ./output_images/binary_images/straight_lines1.jpg "Road preprocessed"
[image4]: ./output_images/warped_undistorted.jpg "Road original warped"
[image5]: ./output_images/left_fitx_right_fitx.jpg "Fit Visual"
[image6]: ./output_images/test4_final_lane.jpg "Output"
[image7]: ./output_images/chessboard_corners/calibration12.jpg "Chessboard corners"
[video1]: ./processed_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Project location

Here is a [link](https://github.com/ahubi/CarND-Advanced-Lane-Lines/blob/master/alf.ipynb) to my IPython notebook where you find the source code for the project. In this [directory](https://github.com/ahubi/CarND-Advanced-Lane-Lines/tree/master/output_images) generated images, used in this writeup, are located. Following table describes the content of output_images directory. As input test images from test_images directory were used.

| Directory / File name       | Description   |
|:-------------:|:-------------:|
| binary_images     | Images after applied threshold and gradient pipeline|
| chessboard_corners      | Images used during camera calibration, detected corners drawn on the image      |
| transformed_images     | Images after being binarized and perspectively transformed.     |
| undistorted     | Images after undistortion is applied |
|left_fitx_right_fitx.jpg| Image after being binarized, applied perspective transform and fitted polynomial lines.|
|test4_final_lane.jpg| Image showing final result lane with text information about left and right radius of the lines and final lane curvature. Distance of the left and right line from center of the car and car position relatively to center. Additionally left and right lines are project on the image.|
|undistorted_cb.jpg| Image showing chessboard images before and after undistortion|
|warped_undistorted.jpg|Image showing original test image with source points drawn in red and the same image after applying perspective transform with destination points drawn in red|


#### 1. Camera Calibration and undistortion

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first (function implemented) and second (function executed) code cell of the IPython notebook located in "./alf.ipynb" The function calibrate_camera() does the calibration.

```
ret, mtx, dist, rvecs, tvecs = calibrate_camera()

```
Chessboard images with 9x6 corners are used. Here is one example a chessboard image used during calibration with the corners drawn on it:

![alt text][image7]

For proper calibration it is advised to have a minimum number of pictures, udacity teacher suggests 20. Different pictures (from different angles and positions) must me taken with the camera which is calibrated.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

#### 2. Pipeline (single images)

To demonstrate this step, I will describe how I apply the process of creating binary thresholded image. Below is an undistorted test image:

![alt text][image2]

#### Create thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image (the code for creating a thresholded binary image is located in the first code cell of my notebook, function pipeline()) Threshholding ranges are passed into function, following values are used:

```
def pipeline(img, s_thresh=(150, 255), sx_thresh=(20, 200),l_thresh=(50,255)):
```
To achieve better results the images is converted to HLS color space in the first step. After that individual channels L and S are separated. Sobel in x direction is applied to L channel to accentuate vertical lines (lane lines). Sobel result is scaled and thresholded. Additionally thresholding is applied to L and S channels separately. At the end all three results are combined to one image by 'AND' and 'OR' operation.

```
combined_binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
```

Below is an image which was processed by the function pipeline. Compared to the undistorted colored image shown above one can see that most of the information is removed, but the important lane lines are still clearly visible.

![alt text][image3]

#### 3. Perspective transform

The code for my perspective transform includes a function called `warp()`, which appears in the first code cell of the IPython notebook.  The `warp()` function takes as inputs an image (`img`) and whether the perspective transform should be done from src to dst or in opposite direction. It returns a warped image provided as argument 'img'.

```
def warp(img,src2dst=True):
```

Source (`src`) and destination (`dst`) points are hardcoded in the function.  I chose to hardcode the source and destination points in the following manner:

```
src = np.float32([[585, 460],[203, 720],[1127, 720],[695, 460]])
dst = np.float32([[320, 0],[320, 720],[960, 720],[960, 0]])

```
Remark: Hardcoding should be ok for this porject since the pipeline operates on constant image size.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial
The source code for detecting lines in the image is located in the fifth notebook cell and is splitted over two functions. The implementation is based on the examples shown during udacity lessons.

The first funciton is used to initially find the left and right line on the image.

```
def detect_start(binary_warped, LL, RL, fname=None):
```

First a histogram is taken of the bottom half of the image

```
histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
```
Find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines

```
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

```
After that repeat identifying the line pixel positions of the image by sliding in 9 steps to the top of the image.

Fit a second order polynomial to each line, using left and right line pixel's coordinates.
```
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

Plotting line pixels, fitted polynomials and the the 9 steps from bottom to top will result in the following picture:

![alt text][image5]


The second function is used when the lines are identified and searches around last position of the lines identified initially with the first function.
```
def detect_next(binary_warped, LL, RL):
```
Both functions take similar parameters, warped image to search line on, left and right line objects to maintain current line detection state.

#### 5. Radius of lane curvature and vehicle position with respect to center.

The source code for radius of the line curvature and position calculation are located in the fifth ans sixth code cell of the notebook. Here is the funciton interface for calculation of line curvature radius:
```
def calculate_curverad(leftx, rightx, ploty):
```
It calculates both radii of left and right line.

The function for calculating the position of each line with respect to the center has the following interface:

```
def line_base_pos(current_fit, ploty):
```

The final lane curvature and the vehicle position with respect to center is calculated in the funciton which maps the lane on the original imaage. This function has the following interface:

```
def draw_lane(undist, warped, LL, RL, fname=None):
```
Radius of the lane curvature is calculated from two line radii.

```
#Calcuate lane radius as a mean of both lines
l = round(LL.radius_of_curvature, 2)
r = round(RL.radius_of_curvature, 2)
medianr = round((l+r)/2, 2)
```
Vehicle position is calculated with positions of left and right line.
```    
#Calcuate vehicle position with right and left line positions
lp = round(LL.line_base_pos, 2)
rp = round(RL.line_base_pos, 2)
offcenter = round((rp - lp)/2,2)
```    
#### 6. Example image for final result plotted back down onto original image.
The implementation for generation such immage is located in the 6th code cell of the the notebook and was described above, draw_lane() function.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)
My video is located in the root of my project directory and is called processed_project_video.mp4.

Here's a [link to my video result.](./processed_project_video.mp4)

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The overall pipeline consists of 5 main steps, all of the steps are marked and described above. Trying the pipeline on the challenge video showed that there is room for improvements and optimizations. During testing on the project video I realized that the bridges with tree shadows and different pavement color are bringing the pipeline into troubles to stay in lane. I applied some sanity checks and tweaked the parameters when evaluating the lines. Please refer to Line class functions radius_in_range(), position_in_range() which are evaluating lines and whether they should be accepted.

Some further investigation could be done in finding a better thresholding process to generate more precise binary images.
Another important area for investigation will be to find out a better line detection algorithm. It might be helpful to calculate radii based on smaller portions of the image. Additionally to that more strict sanity rules could be applied.
