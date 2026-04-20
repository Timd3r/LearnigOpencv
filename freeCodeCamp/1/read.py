import cv2 as cv

#reading an image
# img = cv.imread('Photos/cat.jpg')
# cv.imshow('Cat', img)


#reading a video
#-215:Assertion failed means video or image path is wrong
capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('DogVideo', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()