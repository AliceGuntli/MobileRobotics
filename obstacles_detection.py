#Detection of the corners of the obstacles

#Importations
import cv2
import numpy as np

#Function
def corner_detection(img):
	img_copy = img.copy()

	#Rescaling the image to be seen in the screen
	scale_resize = 1 #1 with images taken by camera, 0.6 with images loaded from computer
	h = img.shape[0]
	w = img.shape[1]
	h_resize = int(scale_resize*h)
	w_resize = int(scale_resize*w)
	imS = cv2.resize(img, (w_resize,h_resize))
		
	#Show the original image
	#cv2.imshow("Original Image", imS)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#Convert to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#grayS = cv2.resize(gray, (w_resize,h_resize))
	#cv2.imshow("Gray Image", grayS)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#Shi-Tomasi method - Play with the parameters depending on the image
	#MinDistance of 80 with images taken with camera
	maxCorners = 0 #Return all the corners detected
	qualityLevel = 0.15 #Level of quality of corner desired - to avoid detect false corners
	minDistance = 20 #Minimum Euclidean distance between two corners - assume a certain size of the obstacles
	corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)

	#Draw red circle on the corners
	corners = np.int0(corners)
	#for i in corners:
	#	x,y = i.ravel()
	#	cv2.circle(img_copy,(x,y),5,(0,0,255),-1)
	

            
	#Show the result
	#img_copyS = cv2.resize(img_copy, (w_resize,h_resize))
	#cv2.imshow("After corner detection", img_copyS)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	return corners


