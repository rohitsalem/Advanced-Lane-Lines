import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import imageio


## camera Calibrate function
def camera_calibrate():
    # collecting all the camera calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    world_points = []  # to store 3D world points also called as object points
    img_points = []   # to store 2D image points

    wrld_points = np.zeros((6*9,3), np.float32)
    wrld_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for file_name in images:
        img = mpimg.imread(file_name)
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # converting into grayscale

        # get chessboard corners
        ret,corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True: # it the corners are found and returned
            img_points.append(corners)
            world_points.append(wrld_points)


    # calibrating the camera using world and image points
    rt, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, (h,w), None, None)

    return mtx, dist

## Function to visualize the images after calibration
def visualize_camera_calibrate(vis=True):
    if (vis):
        mtx, dist  = camera_calibrate()
        image = mpimg.imread('camera_cal/calibration3.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting into grayscale
        # get chessboard corners
        ret,corners = cv2.findChessboardCorners(gray, (9,6), None)
        # draw corners on the image
        image_corners = cv2. drawChessboardCorners(image, (9,6), corners, ret)
        # undistort the image
        image_undistort = cv2.undistort(image, mtx, dist, None, mtx)

        plt.figure(1)
        plt.title("Distorted image with chessboard corners")
        plt.imshow(image_corners)
        plt.savefig("output_images/camera_calibration_distorted.jpg")
        plt.figure(2)
        plt.title("Undistorted image with chessboard corners")
        plt.imshow(image_undistort)
        plt.savefig("output_images/camera_calibration_undistorted.jpg")
        plt.show()

## Function to Visulaize undistorted images
def visualize_undistorted_image(image,vis=True,):
    if (vis):
        mtx, dist = camera_calibrate()
        image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
        plt.figure(1)
        plt.title('Raw image')
        plt.imshow(image)
        plt.savefig("output_images/distorted_image.jpg")
        plt.figure(2)
        plt.title('Undistorted Image')
        plt.imshow(image_undistorted)
        plt.savefig("output_images/undistorted_image.jpg")
        plt.show()


def gradient_thresholding(image, sx_thresh=(20, 100)):
    # convert to HLS color space, separate s channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]

    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) #take x derivative
    sobelx_abs = np.absolute(sobelx) #absolute x derivative
    sobel_scaled = np.uint8(255*sobelx_abs/np.max(sobelx_abs))

    # Thresholding x gradient
    s_binary = np.zeros_like(sobel_scaled)
    s_binary[(sobel_scaled >= sx_thresh[0]) & (sobel_scaled <=sx_thresh[1])] = 1

    binary = np.zeros_like(s_binary)
    binary[(s_binary == 1)]=1

    return binary

## Function to visualize gradient_thresholding
def visualize_gradient_thresholding(image,vis=True):
    if (vis):
        mtx, dist = camera_calibrate()
        image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
        binary = gradient_thresholding(image_undistorted)
        plt.figure(1)
        plt.title('Binary Image')
        plt.imshow(binary,cmap='gray')
        plt.savefig("output_images/binary.jpg")
        plt.show()


def warper(image, mtx, dist, showROI=True, showWarped = True):
    h, w = image.shape[:2]

    # source points depict the region of interest
    source_points = np.float32([[(w/8), h], [(w/8)*7, h], [(w/8)*3.7, h/1.6], [(w/8)*4.3, h/1.6]])
    dest_points = np.float32([[(w/5), h], [(w/5)*4, h], [(w/5), 0], [(w/5)*4, 0]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    Minv = cv2.getPerspectiveTransform(dest_points, source_points)
    warped_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # to show the region of interest
    if(showROI == True):
        c1 = plt.Circle(source_points[0], 5, color=(0, 0, 1))
        c2 = plt.Circle(source_points[1], 5, color=(0, 0, 1))
        c3 = plt.Circle(source_points[2], 5, color=(0, 0, 1))
        c4 = plt.Circle(source_points[3], 5, color=(0, 0, 1))

        # Open new figure
        fig = plt.figure(figsize=(15,10))
        # In figure, Image as background
        plt.imshow(image, cmap='gray')
        # Add the circles to figure as subplots
        fig.add_subplot(111).add_artist(c1)
        fig.add_subplot(111).add_artist(c2)
        fig.add_subplot(111).add_artist(c3)
        fig.add_subplot(111).add_artist(c4)
        plt.title("Region of Interest is shown by blue dots)")
        plt.savefig("output_images/region_of_interest.jpg")
        plt.show()

    if(showWarped==True):
        plt.imshow(warped_image,cmap='gray')
        plt.title("warped_image")
        plt.savefig("output_images/warped_image.jpg")
        plt.show()
    return warped_image, M, Minv


def detect_lane(warped, xm_per_pix, first_image = False, abnormal_threshold=200):
    global left_fit, right_fit
    global leftx, lefty, rightx, righty
    global previous_left_fitx, previous_right_fitx, previous_out_img
    global abnormal_count
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[(int)(binary_warped.shape[0]/2):,:], axis=0)
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if(first_image == True) or (abnormal_count > abnormal_threshold):
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Update previous values for the first time
    if (first_image):
        previous_left_fitx = left_fitx.copy()
        previous_right_fitx = right_fitx.copy()
        previous_out_img = out_img.copy()

    # count the number of abnormal pixels
    abnormal_count = 0
    for i in range(len(left_fitx)):
        distance = (right_fitx[i] - left_fitx[i]) * xm_per_pix
        if (distance <= 0) or (distance > 4.2) or (distance < 3.2):
            abnormal_count += 1
    # some part of this code is inspired from github.com/akshatjain's project
    # If the number of abnormal pixels exceeds a threshold then use the previousiously detected line
    if (abnormal_count > abnormal_threshold):
        left_fitx = previous_left_fitx.copy()
        right_fitx = previous_right_fitx.copy()
        out_img = previous_out_img.copy()
    else:
        previous_left_fitx = left_fitx.copy()
        previous_right_fitx = right_fitx.copy()
        previous_out_img = out_img.copy()

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fitx, right_fitx, ploty

def radius_of_curvature(xm_per_pix, ym_per_pix):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    radius_of_curvature = min(left_curverad, right_curverad)

    return radius_of_curvature

def vehicle_position(xm_per_pix, ym_per_pix):

    center = ((right_fitx[-1]*xm_per_pix)-(left_fitx[-1]*xm_per_pix))/2
    center_actual = 3.7*0.5
    diff = abs(center-center_actual)

    if(left_fitx[-1]<(1280-right_fitx[-1])):
        direction = 'right'
    elif(left_fitx[-1] > (1280 - right_fitx[-1])):
        direction = 'left'
    else:
        direction = 'center'

    return diff, direction

def output(undistorted_image):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (test_image.shape[1], test_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    return result


##### Pipeline:
### Calibrating camera
mtx, dist  = camera_calibrate()

## Visualize_camera_calibrate on sample image, Change the flag to False to skip Visualizing camera_calibrate images
# visualize_camera_calibrate(True)

#Loading a test image
test_image = mpimg.imread('test_images/test1.jpg')
### Undistorting the images
## Visulaize undistored images on test_image , Change the flag to True to Visualize undistorted_image
visualize_undistorted_image(test_image, False)

### gradient_thresholding, Change the flag to True to skip Visualize Binary image
visualize_gradient_thresholding(test_image, False)

binary_image = gradient_thresholding(test_image)
## change the flags to True to visualize the Region of interest and warped images
binary_warped, M, Minv = warper(binary_image, mtx, dist, showROI=False, showWarped=False)

# Define conversions in x and y from pixels space to meters
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ym_per_pix = 30/720 # meters per pixel in y dimension

#Lane detection
lane_detected, left_fitx, right_fitx, ploty = detect_lane(binary_warped, xm_per_pix, True)
visualize_lane = False
if (visualize_lane):
    plt.imshow(lane_detected)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig("output_images/warped_detected_lane.jpg")
    plt.show()

# Get radius of curvature
# radius_of_curvature = radius_of_curvature(xm_per_pix, ym_per_pix)

## TO get the video
# # Define conversions in x and y from pixels space to meters
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ym_per_pix = 30/720 # meters per pixel in y dimension

# Test on Video
first_image = True
filename = 'project_video.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
fps = vid.get_meta_data()['fps']
writer = imageio.get_writer('project_out.mp4',fps=fps)
print("Testing on video, Please wait ..")
for i,im in enumerate(vid):
    undistort = cv2.undistort(im, mtx, dist, None, mtx)
    binary = gradient_thresholding(undistort)
    binary_warped, M, Minv = warper(binary, mtx, dist, showROI=False, showWarped=False)
    lane_detected, left_fitx, right_fitx, ploty = detect_lane(binary_warped, xm_per_pix, first_image)
    if(first_image):
        first_image = False
    radius = radius_of_curvature(xm_per_pix, ym_per_pix)
    difference, direction = vehicle_position(xm_per_pix, ym_per_pix)
    output_img = output(undistort)

    cv2.putText(output_img,"Radius of curvature: {0:.2f} meters".format(radius), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.putText(output_img,"Vehicle is {0:.2f} m {1} of center".format(difference, direction), (25,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    writer.append_data(output_img[:,:,:])

writer.close()
## To Display on test image
visualize_test_image = False
if (visualize_test_image):
    radius_of_curvature = radius_of_curvature(xm_per_pix, ym_per_pix)
    first_image = True
    undistort = cv2.undistort(test_image, mtx, dist, None, mtx)
    binary = gradient_thresholding(undistort)
    binary_warped, M, Minv = warper(binary, mtx, dist, showROI=False, showWarped=False)
    lane_detected, left_fitx, right_fitx, ploty = detect_lane(binary_warped, xm_per_pix, first_image)

    if(first_image):
        first_image = False
    difference, direction = vehicle_position(xm_per_pix, ym_per_pix)
    output = output(undistort)
    plt.figure()
    plt.imshow(output)
    plt.text(25, 40, "Radius of curvature: {0:.2f} meters".format(radius_of_curvature))
    plt.text(25, 80, "Vehicle is {0:.2f} m {1} of center".format(difference, direction))
    plt.savefig("output_images/output.jpg")
    plt.show()
