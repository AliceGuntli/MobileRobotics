#Main of computer vision

#Importations
import cv2
import numpy as np
from tools import load
from start_and_goal import start_goal
from obstacles_detection import corner_detection

#Main
img = load()
cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
start, goal = start_goal(img)
corners = corner_detection(img)

#Check that Thymio is not is the list of corners of obstacles
#If it is, remove it
thymio = np.array((start[0], start[1]))
for i in range(len(corners)-1) :
	if (np.linalg.norm(thymio-corners[i,:]) < 55) :
		corners = np.delete(corners, i, 0)

output = img.copy()
for i in corners:
		x,y = i.ravel()
		cv2.circle(output,(x,y),5,(0,0,255),-1)
		
cv2.imshow("Only corners of obstacles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()	

print("Start", start)
print("Goal", goal)
print("Corners of the obstacles", corners)
