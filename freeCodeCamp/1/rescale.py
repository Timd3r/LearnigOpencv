import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    #images, video and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #live video
    capture.set(3,width)
    capture.set(4,height)

img = cv.imread('Photos/cat_large.jpg')
cv.imshow('Cat Resized', rescaleFrame(img, scale=.2))
cv.waitKey(0)

# capture = cv.VideoCapture('Videos/dog.mp4')

# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescaleFrame(frame, scale=.2)
#     cv.imshow('DogVideo', frame)
#     cv.imshow('DogVideo Resized', frame_resized)

#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()