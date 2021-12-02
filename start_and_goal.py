#Function that returns the centroid of the start and the centroid of the goal

#Importations
import cv2
import numpy as np

#Function
def start_goal(img):
	blur = cv2.GaussianBlur(img,(1,1),0)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# apply connected component analysis to the thresholded image
	output = cv2.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv2.CV_32S)
	(numLabels, labels, stats, centroids) = output
	output = img.copy()

	# loop over the number of unique connected component labels
	thymio = []
	for i in range(0, numLabels):
		# extract the connected component statistics and centroid for
		# the current label
		x = stats[i, cv2.CC_STAT_LEFT]		#Coordinate x of the first point
		y = stats[i, cv2.CC_STAT_TOP]		#Coordinate y of the first point
		w = stats[i, cv2.CC_STAT_WIDTH]		#Width
		h = stats[i, cv2.CC_STAT_HEIGHT]	#Height	
		area = stats[i, cv2.CC_STAT_AREA]	#Area
		(cX, cY) = centroids[i]				#Centroids
	
	
		# ensure the width, height, and area are all neither too small nor too big
		keepWidth = w > 10 and w < 2000
		keepHeight = h > 10 and h < 200
		keepArea = area > 50  #way to identify the goal and the start ! 
	
		# ensure the connected component we are examining passes all
		# three tests
		if all((keepWidth,keepHeight, keepArea)):
			
			#Detect the two triangles representing the thymio from their area
			if ((area < 650) and (area > 500)):
				thymio.append(int(cX))
				thymio.append(int(cY))
				#cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
				#cv2.circle(output, (int(cX), int(cY)), 4, (255, 0, 0), -1)

			
			#Detect the goal from its area
			if ((area < 1900) and (area > 1400)):
				goal = [cX, cY]
				#cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
				cv2.circle(output, (int(cX), int(cY)), 4, (0, 255, 0), -1)
	
	#Start (Thymio) as the mean of the two triangles
	start = []
	start.append(int((thymio[0]+thymio[2])/2))
	start.append(int((thymio[1]+thymio[3])/2))
	cv2.circle(output, (start[0], start[1]), 8, (255, 0, 0), -1)
	
	#Front with the value of the pixel
	if gray[thymio[1], thymio[0]] > gray[thymio[3], thymio[2]] :
		front = [thymio[2], thymio[3]]
	else :
		front = [thymio[0], thymio[1]]
	

	cv2.circle(output, (front[0], front[1]), 6, (0, 0, 255), -1)
	
	# show our output image and connected component mask
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	
	#start[0] = int(start[0])
	#start[1] = int(start[1])

	goal[0] = int(goal[0])
	goal[1] = int(goal[1])

	return start, goal, front


