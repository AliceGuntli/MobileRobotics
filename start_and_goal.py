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
			
			#Detect the start from its area
			if ((area < 500) and (area > 450)):
				start = [cX, cY]
				cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
				cv2.circle(output, (int(cX), int(cY)), 4, (255, 0, 0), -1)
				#cv2.circle(output, (int(x+w), int(y+h)), 10, (0, 255, 0), -1)
				p1 = np.array((x,y))
				p2 = np.array((x+w,y))
				p3 = np.array((x,y+h))
				p4 = np.array((x+w,y+h))
				start_np = np.array((cX, cY))

				front = []
				front.append(p1[0])
				front.append(p1[1])
				print("front", front)
				dist = np.linalg.norm(start_np-p1)
				
				if np.linalg.norm(start_np-p2) < dist :
					front[0] = p2[0]
					front[1] = p2[1]
					dist = np.linalg.norm(start_np-p2)
				
				if np.linalg.norm(start_np-p3) < dist :
					front[0] = p3[0]
					front[1] = p3[1]
					dist = np.linalg.norm(start_np-p3)
					
				if np.linalg.norm(start_np-p4) < dist :
					front[0] = p4[0]
					front[1] = p4[1]
					dist = np.linalg.norm(start_np-p4)
				
				print("front", front)
				x1 = front[0]
				y1 = front[1]
				cv2.circle(output, (x1,y1), 8, (0, 0, 255), -1)
			
			#Detect the goal from its area
			if ((area < 1900) and (area > 1700)):
				goal = [cX, cY]
				cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
				cv2.circle(output, (int(cX), int(cY)), 4, (0, 255, 0), -1)
	
	# show our output image and connected component mask
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	
	start[0] = int(start[0])
	start[1] = int(start[1])

	goal[0] = int(goal[0])
	goal[1] = int(goal[1])

	return start, goal, front


