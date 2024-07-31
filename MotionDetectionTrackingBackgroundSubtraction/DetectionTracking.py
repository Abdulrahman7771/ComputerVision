import numpy as np
import cv2 as cv
from tracker import *

result = cv.VideoWriter("out.mp4",cv.VideoWriter_fourcc(*'XVID')
                        ,20, (250,250))
cap = cv.VideoCapture("highway.mp4")

tracker = EuclideanDistTracker()

ObjectDetector = cv.createBackgroundSubtractorMOG2(100,50)

while cap.read():
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        roi = frame[340: 720,500: 800] 
        
        mask = ObjectDetector.apply(roi)
        _,mask = cv.threshold(mask,244,255,cv.THRESH_BINARY)
        contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
        detections = []
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area>100:
                x,y,w,h = cv.boundingRect(contour)
                detections.append([x, y, w, h])              
    
        Bids = tracker.update(detections)
        for Bid in Bids:
            x,y,w,h,Id = Bid          
            cv.putText(roi, str(Id), (x, y)
                        ,cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
        cv.imshow("roi", roi)
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask)
        
        result.write(frame)             
    
    key = cv.waitKey(30)
    if key == 27:
        break        

result.release()
cap.release()

cv.destroyAllWindows()