#Test file for computer vision

#Some importation useful for image processing (not keep all)
import cv2
import time

import numpy as np


import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
from tqdm import tqdm

import IPython.display as Disp
from ipywidgets import widgets

#Take a picture with the camera and save it under the name "frame.jpg"
def take_frame():
	# 1.creating a video object
	video = cv2.VideoCapture(1) 
	# 2. Variable
	a = 0
	# 3. While loop
	while True:
		a = a + 1
		# 4.Create a frame object
		check, frame = video.read()
		# Converting to grayscale
		#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		# 5.show the frame!
		cv2.imshow("Capturing",frame)
		# 6.for playing 
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
	# 7. image saving
	showPic = cv2.imwrite("frame.jpg",frame)
	print(showPic)
	# 8. shutdown the camera
	video.release()
	cv2.destroyAllWindows() 

def load():
	img = cv2.imread('images/formes.png', cv2.IMREAD_COLOR)
	#img = cv2.imread('frame.jpg', cv2.IMREAD_COLOR)
	# If the image path is wrong, the resulting img will be none
	if img is None:
		print('Open Error')
	else:
		print('Image Loaded')
	
	return img

def corner_detection(img):
	img_copy = img.copy()

	#Rescaling the image to be seen in the screen
	scale_resize = 0.5 #1 with images taken by camera, 0.6 with images loaded from computer
	h = img.shape[0]
	w = img.shape[1]
	h_resize = int(scale_resize*h)
	w_resize = int(scale_resize*w)
	imS = cv2.resize(img, (w_resize,h_resize))

	#Show the original image
	cv2.imshow("Original Image", imS)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	grayS = cv2.resize(gray, (w_resize,h_resize))
	cv2.imshow("Gray Image", grayS)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Shi-Tomasi method - Play with the parameters depending on the image
	#MinDistance of 80 with images taken with camera
	maxCorners = 0 #Return all the corners detected
	qualityLevel = 0.2 #Level of quality of corner desired - to avoid detect false corners
	minDistance = 250 #Minimum Euclidean distance between two corners - assume a certain size of the obstacles
	corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)

	#Draw red circle on the corners
	corners = np.int0(corners)
	for i in corners:
		x,y = i.ravel()
		cv2.circle(img_copy,(x,y),5,(0,0,255),-1)
	img_copyS = cv2.resize(img_copy, (w_resize,h_resize))

            
	#Show the result
	cv2.imshow("After corner detection", img_copyS)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return corners

def remove_shdw(img)
	shdw_img = img.copy()

	rgb_planes = cv2.split(shdw_img)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
		dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
		bg_img = cv2.medianBlur(dilated_img, 21)
		diff_img = 255 - cv2.absdiff(plane, bg_img)
		norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		result_planes.append(diff_img)
		result_norm_planes.append(norm_img)

	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	
	cv2.imshow("After shadow removal", result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return result_norm


	
#Main
take_frame()
#img = load()
#corners = corner_detection(img)

#List containing the coordinates of the corners - To use it later
#print(corners)
#print(len(corners))

#Detection of triangle
#coordinates_triangle = triangle_detection(img);
#print(coordinates_triangle)

#To do : just add a certain value to the corners to grow the obstacles
