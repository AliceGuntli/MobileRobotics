#Try to identify a triangle

#Some importation useful for image processing (not keep all)
import cv2
import time

import numpy as np
#import imutils

import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
from tqdm import tqdm

import IPython.display as Disp
from ipywidgets import widgets

#Load an image
def load():
	img = cv2.imread('frame.jpg', cv2.IMREAD_COLOR)
	#img = cv2.imread('frame.jpg', cv2.IMREAD_COLOR)
	# If the image path is wrong, the resulting img will be none
	if img is None:
		print('Open Error')
	else:
		print('Image Loaded')
	
	return img

#First idea
def triangle_detection(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((4, 4), np.uint8)
	dilation = cv2.dilate(gray, kernel, iterations=1)

	blur = cv2.GaussianBlur(dilation, (11, 11), 0)


	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

	# Now finding Contours         ###################
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cnts = imutils.grab_contours(cnts)
	for cnt in cnts:
		# [point_x, point_y, width, height] = cv2.boundingRect(cnt)
		#approx = cv2.approxPolyDP(cnt, 0.07 * cv2.arcLength(cnt, True), True)
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
		if len(approx) == 3:
			print("Triangle")
			coordinates.append([cnt])
			cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)

	cv2.imshow("Detection of triangle",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return coordinates

#Second idea
img = load()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv2.CV_32S)
(numLabels, labels, stats, centroids) = output


# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	print(w)
	print(h)
	print(area)
	
	output = img.copy()
	
	# ensure the width, height, and area are all neither too small
	# nor too big
	keepWidth = w > 10 and w < 500
	keepHeight = h > 10 and h < 500
	keepArea = area > 400  #way to identify the goal and the start ! 
	
	# ensure the connected component we are examining passes all
	# three tests
	if all((keepWidth,keepHeight, keepArea)):
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
		cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

		if ((area < 600) and (area > 500)):
			start = [(cX, cY)]
			cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
			cv2.circle(output, (int(cX), int(cY)), 4, (255, 0, 0), -1)
		if ((area < 1900) and (area > 1700)):
			goal = [(cX, cY)]
			cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
			cv2.circle(output, (int(cX), int(cY)), 4, (0, 255, 0), -1)

		# show our output image and connected component mask
		cv2.imshow("Output", output)
		cv2.waitKey(0)

#Detection of triangle
coordinates_triangle = triangle_detection(img);
print(coordinates_triangle)

