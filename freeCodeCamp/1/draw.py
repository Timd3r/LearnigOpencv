import cv2 as cv
import numpy as np

#blank image
blank = np.zeros((500, 500, 3), dtype='uint8')
#cv.imshow('Blank', blank)

#1. Paint the image a certain color
# blank[:] = 0,255,0
blank[200:300, 300:400] = 0,255,0

#2. Draw a rectangle
cv.rectangle(blank, (0,0), (250,250), (0,0,255), thickness=5)

#3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255,0,0), thickness=-1)

#4. Draw a line
cv.line(blank, (250,250), (300,350), (255,255,255), thickness=3)

#5. Write text
cv.putText(blank, 'Hello World', (5, 40), cv.FONT_HERSHEY_TRIPLEX, 1.2, (255,255,255), thickness=2)

cv.imshow('drawing', blank)

cv.waitKey(0)