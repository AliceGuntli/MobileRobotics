#Main of computer vision

#Importations of librairies
import cv2
import numpy as np
import pyvisgraph as vg
from tools import *
from start_and_goal import *
from obstacles_detection import *
from intersection import *


#Main
img = load()
cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
corners = corner_detection(img)
start, goal, front = start_goal(img)


#Remove points that are part of the Thymio from the list of corners of obstacles
thymio = np.array((start[0], start[1]))
to_be_del = []	#Index of element to delete (otherwise problem because len of corners changes during function)

#Store the elements to be deleted
for i in range(len(corners)) :
	if (np.linalg.norm(thymio-corners[i,:,:]) < 45) :
		to_be_del.append(i)

#Sort the list reverse order to avoid problem of changing length of list
to_be_del.sort(reverse = True)
for i in to_be_del :
		corners = np.delete(corners, i, 0)

#Draw circles around the corners and show image (just for check)		
output = img.copy()
for i in corners:
		x,y = i.ravel()
		cv2.circle(output,(x,y),5,(0,0,255),-1)

#plt.imshow(output)
cv2.imshow("Only corners of obstacles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Some print for check
#print("Start", start)
#print("Goal", goal)
#print("Corners of the obstacles", corners)

#Maybe possible to have only one simpler function here
#Return list of elements composed of two points each, representing diagonal of obstacle (by checking the value of the pixels between the two points)
possible_obstacles = draw_lines(img, corners) 	#to be done : change this function ? 
#print(possible_obstacles)
#Return a list of elements composed of four points each, representing an entire obstacle (by checking the intersection)
obstacles = real_obstacles(possible_obstacles)
#print("obstacles")
#print(obstacles)

#Compute the shortest path with the library
polys = []
start_vg = vg.Point(start[0], start[1])
goal_vg = vg.Point(goal[0], goal[1])
for i in range(len(obstacles)) :
		polys.append([vg.Point(obstacles[i][0], obstacles[i][1]), vg.Point(obstacles[i][2], obstacles[i][3]), vg.Point(obstacles[i][4], obstacles[i][5]), vg.Point(obstacles[i][6], obstacles[i][7])])
g = vg.VisGraph()
g.build(polys)
shortest = g.shortest_path(start_vg, goal_vg)
print(shortest)

#Put a red circle on points of the path (to check)
path = img.copy()
for point in shortest :
	cv2.circle(path,(int(point.x),int(point.y)),5,(0,0,255),-1)
	
cv2.imshow("Path", path)
cv2.waitKey(0)
cv2.destroyAllWindows()
