import numpy as np
import cv2

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
        img = cv2.imread('./assets/cs.jpg')
        cv2.imshow('image', img)
        cv2.waitKey(0)

if __name__ == "__main__":
    PROCESS = Process()
    PROCESS.main()