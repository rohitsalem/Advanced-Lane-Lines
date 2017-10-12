
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

[image1]: ./output_images/camera_calibration_distorted.jpg "Distorted image chessboard"
[image2]: ./output_images/camera_calibration_undistorted.jpg "UnDistorted image chessboard"
[image3]: ./output_images/distorted_image.jpg "Distorted test image"
[image4]: ./output_images/undistorted_image.jpg "Undistorted test image"
[image5]: ./output_images/binary.jpg "Binary image"
[image6]: ./output_images/region_of_interest.jpg "ROI"
[image7]: ./output_images/warped_image.jpg "warped image"
[image8]: ./output_images/warped_detected_lane.jpg "warped deteced lane"
[image9]: ./output_images/output.jpg "output"


### Camera Calibration

1. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `wrld_points` is just a replicated array of coordinates, and `world_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `wrld_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

![alt text][image2]


### Pipeline 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3] 
![alt text][image4]

I used gradient thresholds to generate a binary image . Here's an example of my output for this step:
![alt text][image5]


The code for my perspective transform includes a function called `warper()` which appears in the `p4.py` line 106.  The `warper()` function takes as inputs an image (`img`), and calibration matrices and source (`src`) and destination (`dst`) points are defined in the function itself.  I chose the hardcode the source and destination points in the following manner:

```python
source_points = np.float32([[(w*0.125), h], [w*0.9, h], [(w/8)*3.7, (h/8)*5], [(w/8)*4.6, (h/8)*5]])
dest_points = np.float32([[(w/5), h], [(w/5)*4, h], [(w/5), 0], [(w/5)*4, 0]])
```

I verified that my perspective transform was working as expected by drawing the `source` and `dest` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

![alt text][image7]

I used the technique suggested in the class, the sliding window and histogram technique to identify all the non-zero pixels along the lane. The output is:

![alt text][image8]

The functions for calculating the radius of curvature (lines 270-284) and vehicle position (lines 286-299) can be found in the p4.py 
Final output on a test image:

![alt text][image9]

The Project output on the video can be found [here](project_out.mp4)
My code still fails sometime while using the challenge video, I look forward to improve the detect_lane function along with fine tuning of the thresholding. 

