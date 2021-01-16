import cv2 as cv
import numpy as np
import pyautogui

cam = cv.VideoCapture(0)

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([40,225,225])

lower_green = np.array([50,100,100])
upper_green = np.array([80,225,225])

while(True):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)

    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    mask = np.zeros_like(frame)
    mask[50:350, 50:350] = [255,255,255]
    
    image_roi = cv.bitwise_and(image_smooth, mask)

    cv.rectangle(frame, (50,50), (350,350),(0,0,255),2)
    cv.line(frame, (150,50), (150,350),(0,0,255),1)
    cv.line(frame, (250,50), (250,350),(0,0,255),1)
    cv.line(frame, (50,150), (350,150),(0,0,255),1)
    cv.line(frame, (50,250), (350,250),(0,0,255),1)
    
    
    image_hsv = cv.cvtColor(image_roi, cv.COLOR_BGR2HSV)

    image_threshold = cv.inRange(image_hsv, lower_yellow, upper_yellow)

    contours, heirarchy = cv.findContours(image_threshold, \
                                                           cv.RETR_TREE, \
                                                           cv.CHAIN_APPROX_NONE) \

    if(len(contours)!=0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
##        x_bound, y_bound, w_bound, h_bound = cv.boundingRect(cnt)
##        cv.rectangle(frame, (x_bound, y_bound), (x_bound + w_bound, y_bound + h_bound),(255,0,0))
        M = cv.moments(cnt)
        if(M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(frame, (cx,cy),4,(0,255,0),-1)

            if cx < 150:
                dist_x = -20
            elif cx > 250:
                dist_x = 20
            else:
                dist_x = 0

            if cy < 150:
                dist_y = -20
            elif cy > 250:
                dist_y = 20
            else:
                dist_y = 0
            pyautogui.moveRel(dist_x, dist_y, duration=0.25)

            image_threshold_green = cv.inRange(image_hsv, lower_green, upper_green)

            contours_green, heirarchy = cv.findContours(image_threshold_green, \
                                                           cv.RETR_TREE, \
                                                           cv.CHAIN_APPROX_NONE)
            if(len(contours_green)!=0):
                pyautogui.click()
                cv.waitKey(1000)
                
                    
    cv.imshow('Frame', frame)
    key = cv.waitKey(100)
    if key == 27:
                     break
cam.release()
