#Intersections
#Importations
import cv2
import numpy as np
from tools import *

#Function to compute line between the corners and with the color depending on the value of the pixels 
def compute_line(x1,y1, x2, y2, img) :
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pt_a = np.array([x1, y1])
    pt_b = np.array([x2, y2])
    count_red = 0
    count_green = 0
    for p in np.linspace(pt_a, pt_b, 50):
        x,y = p.ravel()
        x = int(x)
        y = int(y)
        if img_gray[y,x] < 100 :
            cv2.circle(img, (x,y), 4, (0, 0, 255), -1)
            count_red += 1
        else :
            cv2.circle(img, (x,y), 4, (0, 255, 0), -1)
            count_green += 1
        
    return count_red, count_green
    
#Draw lines
def draw_lines(img, corners) :
	h = img.shape[0]
	w = img.shape[1]
	img2 = img.copy()
	img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	possible_obstacles = []
	#free_path = []

	for i in range (0,len(corners)) :
		for j in range(i+1, len(corners)) :
			if i != j :
				x1,y1 = corners[i].ravel()
				x2,y2 = corners[j].ravel()
				red, green = compute_line(x1,y1,x2,y2,img2)
				if red > 48 :
					possible_obstacles.append([x1,y1,x2,y2])
				#if green < 40 :
					#free_path.append([x1,y1,x2,y2])

                
	#print("Obstacles ", possible_obstacles)
	#print("Free path", free_path)

	#Show the result
	#img2S = cv2.resize(img2, (w,h))
	#cv2.imshow("Line between two points", img2S)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return possible_obstacles

#Draw lines that are obstacles
#To see which lines are in the obstacle category
def obstacles_cat() :

	img3 = img.copy()
	
	for i in range(0,len(possible_obstacles)) :
		x1 = possible_obstacles[i][0]
		y1 = possible_obstacles[i][1]
		x2 = possible_obstacles[i][2]
		y2 = possible_obstacles[i][3]
		red, green = compute_line(x1,y1,x2,y2,img3)

	#Show the result
	img3S = cv2.resize(img3, (w_resize,h_resize))
	cv2.imshow("Draw only lines that are obstacles", img3S)
	cv2.waitKey(0)


cv2.destroyAllWindows()

def real_obstacles(possible_obstacles) :
	obstacles = []
	for i in range (0,len(possible_obstacles)) :
		x1 = possible_obstacles[i][0]
		y1 = possible_obstacles[i][1]
		x2 = possible_obstacles[i][2]
		y2 = possible_obstacles[i][3]
		p1 = Point(x1,y1)
		q1 = Point(x2,y2)
		for j in range (i, len(possible_obstacles)) :
			x3 = possible_obstacles[j][0]
			y3 = possible_obstacles[j][1]
			x4 = possible_obstacles[j][2]
			y4 = possible_obstacles[j][3]
			p2 = Point(x3,y3)
			q2 = Point(x4,y4)
			if ((p1.x != p2.x) and (p1.y != p2.y) and (p1.x != q2.x) and (p1.y != q2.y) and (q1.x != p2.x) and (q1.y != p2.y) and (q1.x != q2.x) and (q1.y != q2.y)) :
				if doIntersect(p1,q1,p2,q2) :
					element = possible_obstacles[i] + possible_obstacles[j] 
					obstacles.append(element)

	return obstacles      
