import cv2
import numpy as np
import imutils


class Transform:
    def __init__(self):
        self.bg_sub = None

    def init_SubBackground(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()

    def removeNoise(self, frame, iterations = 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for i in range(iterations):
            #filtered_mask = cv2.bilateralFilter(frame, 3, 175, 175)
            filtered_mask = cv2.erode(frame, kernel, iterations=1)
            #filtered_mask = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        _, thresh_mask = cv2.threshold(filtered_mask, 0.6 * filtered_mask.max(), 255, cv2.THRESH_BINARY)
        return thresh_mask

    def subBackground(self, frame, clean = 1):
        if self.bg_sub is None:
            self.init_SubBackground()
        fg_mask = self.bg_sub.apply(frame)
        fg_mask_clean = self.removeNoise(fg_mask, clean)
        return fg_mask_clean

    def drawContour(self, frame, contours):
        if len(contours) is not 0:
            human_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            (x, y, w, h) = cv2.boundingRect(human_contour)
            middle = (x + int(w / 2), y + int(h / 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.rectangle(frame, (middle[0] - 2, middle[1] - 2), (middle[0] + 2, middle[1] + 2), (255, 0, 0), -1)
            return middle

    def findContour(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def HOG(self, frame):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        frame = imutils.resize(frame, width=min(400, frame.shape[1]))

        rects, weight = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        print(len(rects))
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return frame

    def handleLines(self, frame, amount_lines, middle):
        lines = []
        height, width, _ = frame.shape
        for i in range(amount_lines):
            p1 = (int(width / amount_lines) * i, 0)
            p2 = (int(width / amount_lines) * i, height)
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            lines.append([p1, p2])

        for line in lines:
            if  line[0][0] - 5 < middle[0] < line[0][0] + 5:
                cv2.line(frame, line[0], line[1], (0, 255, 0), 2)


t = Transform()
cap = cv2.VideoCapture('img/walking.mp4')

while 1:
    ret, frame = cap.read()
    fg_mask = t.subBackground(frame, 5)
    middle = t.drawContour(frame, t.findContour(fg_mask))
    if middle is not None:
        t.handleLines(frame, 10, middle)
    #cv2.imshow('test', fg_mask)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()