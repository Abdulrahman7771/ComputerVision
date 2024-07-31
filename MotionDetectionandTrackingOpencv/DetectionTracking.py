import numpy as np
import cv2 as cv

cap = cv.VideoCapture("vtest.avi")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc('X','V','I','D')
out = cv.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    if ret:
        diff = cv.absdiff(frame1,frame2)
        gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)
        _,thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        
        contours,_ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h =  cv.boundingRect(contour)     
            if cv.contourArea(contour) < 600:              
                continue    
            cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)              
            cv.putText(frame1, "Status: {}".format('Movement'), (x, y)
                        ,cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
            
        image = cv.resize(frame1, (1280,720))
        out.write(image)
        cv.imshow("feed", frame1)
        
        frame1 = frame2
        ret, frame2 = cap.read()        
        if cv.waitKey(40) == 27:
            break     
cv.destroyAllWindows()#close window

cap.release()#close camera

out.release()#close when write

