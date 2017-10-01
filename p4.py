import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## camera Calibrate function
def camera_calibrate(visualize= True):
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

    if(visualize):
        print(images)
        image = mpimg.imread(images[0])
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting into grayscale
        # get chessboard corners
        ret,corners = cv2.findChessboardCorners(gray, (9,6), None)

        # draw corners on the image
        image_corners = cv2. drawChessboardCorners(image, (9,6), corners, ret)
        # undistort the image
        image_undistort = cv2.undistort(image, mtx, dist, None, mtx)

        plt.figure(1)
        plt.subplot(121)
        plt.title("Distorted image with corners")
        plt.imshow(image_corners)
        plt.subplot(122)
        plt.title("Undistorted image with corners")
        plt.imshow(image_undistort)
        plt.savefig("output_images/camera_calibration.jpg")
        plt.show()

    return mtx, dist

camera_calibrate()
