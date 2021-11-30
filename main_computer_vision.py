#Main of computer vision

#Importations of librairies
import cv2
import numpy as np
import pyvisgraph as vg
from tools import *
from start_and_goal import start_goal
from obstacles_detection import *
from intersection import *


#Main
img = load()
cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
corners = corner_detection(img)
start, goal, front = start_goal(img)


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
#print("Corners of the obstacles", corners)

possible_obstacles = draw_lines(img, corners)
#print(possible_obstacles)
obstacles = real_obstacles(possible_obstacles)
print("obstacles")
print(obstacles)

#Graph shortest path
polys = []
start_vg = vg.Point(start[0], start[1])
goal_vg = vg.Point(goal[0], goal[1])
for i in range(len(obstacles)) :
		polys.append([vg.Point(obstacles[i][0], obstacles[i][1]), vg.Point(obstacles[i][2], obstacles[i][3]), vg.Point(obstacles[i][4], obstacles[i][5]), vg.Point(obstacles[i][6], obstacles[i][7])])

g = vg.VisGraph()
g.build(polys)
shortest = g.shortest_path(start_vg, goal_vg)

print(shortest)
