<<<<<<< HEAD
import cv2
import numpy as np
import imutils


class Transform:
    def __init__(self):
        self.bg_sub = None

    def init_sub_background(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()

    def remove_noise(self, frame, iterations = 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for i in range(iterations):
            #filtered_mask = cv2.bilateralFilter(frame, 3, 175, 175)
            filtered_mask = cv2.erode(frame, kernel, iterations=1)
            #filtered_mask = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        _, thresh_mask = cv2.threshold(filtered_mask, 0.6 * filtered_mask.max(), 255, cv2.THRESH_BINARY)
        return thresh_mask

    def sub_background(self, frame, clean = 1):
        if self.bg_sub is None:
            self.init_sub_background()
        fg_mask = self.bg_sub.apply(frame)
        fg_mask_clean = self.remove_noise(fg_mask, clean)
        return fg_mask_clean


t = Transform()
cap = cv2.VideoCapture('img/walking.mp4')

while 1:
    ret, frame = cap.read()
    fg_mask = t.sub_background(frame, 5)
    cv2.imshow('test', fg_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
=======
>>>>>>> c5394ce0dff49fce9401803813102995510bb49c
