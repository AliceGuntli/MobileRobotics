#Tools 

#Importations
import cv2

#Take picture with camera
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
