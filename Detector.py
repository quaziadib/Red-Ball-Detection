import cv2 as cv
import imutils


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

red_lo_1 = (0, 100, 100)
red_hi_1 = (10, 255, 255)

red_lo_2 = (160, 100, 100)
red_hi_2 = (179, 255, 255)


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    if frame is None:
        break

    frame_resz = rescaleFrame(frame)
    blurred = cv.GaussianBlur(frame_resz, (7, 7), cv.BORDER_DEFAULT)
    col_hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask_1 = cv.inRange(col_hsv, red_lo_1, red_hi_1)
    mask_2 = cv.inRange(col_hsv, red_lo_2, red_hi_2)
    mask = cv.addWeighted(mask_2, 1.0, mask_1, 1.0, 0.0)
    mask = cv.erode(mask, (7, 7), iterations=2)
    mask = cv.dilate(mask, (7, 7), iterations=2)
    contors = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contors = imutils.grab_contours(contors)

    if len(contors) > 0:
        large = max(contors, key = cv.contourArea)
        ((x,y), radius) = cv.minEnclosingCircle(large)
        moms = cv.moments(large)
        try:
            center = (int(moms['m10']/moms['m00']), int(moms['m01']/moms['m00']))
        except ZeroDivisionError:
            pass

        if radius > 10:
            cv.circle(frame_resz, (int(x), int(y)), int(radius),(0, 255, 0), 2)


    cv.imshow("Video", frame_resz)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()


# Reference:
# https://stackoverflow.com/questions/42840526/opencv-python-red-ball-detection-and-tracking
# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
