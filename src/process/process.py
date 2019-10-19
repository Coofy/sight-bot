import numpy as np
import cv2

UPPER_GREEN = [154,169,63] # brightest
LOWER_GREEN = [7,6,1] # darkest

class Process(object):
    def __init__(self):
        self.frames = []

    def main(self):
        # Code below is suitable for capturing video frames
        # cap = cv2.VideoCapture('url')
        # while True:
        #   res, frame = cap.read()
        #   if not res:
        #       break
        # Use frame to manipule frames
        upper_green = np.array([UPPER_GREEN])
        lower_green = np.array([LOWER_GREEN])
        img = cv2.imread('./assets/cs.jpg')

        # Threshold the RGB image to get only green colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, lower_green, upper_green)

        cv2.imshow('image', img)
        cv2.imshow('mask', mask)
        k = cv2.waitKey(0)
        if k == 27:
            return

if __name__ == "__main__":
    PROCESS = Process()
    PROCESS.main()