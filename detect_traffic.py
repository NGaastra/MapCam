import cv2
import math
import numpy as np
import imutils


# Notes
# 180 - 90 - cam angle = floor angle respect to camera
# height gain = tan(angle)
# position = height / height gain

# PROBLEMS ATM
# None

# TODO
# Haar cascade for forklift
# Position fetching


class Feed:
    def __init__(self, feed):
        self.feed = cv2.VideoCapture(feed)

    def read(self):
        ret, read = self.feed.read()
        crop = imutils.resize(read, width=640, height=480)
        return ret, crop

    def set_feed(self, feed):
        self.feed = cv2.VideoCapture(feed)

    def close(self):
        return self.feed.close()


class Cascade:
    def __init__(self, cascade):
        self.cascade = cv2.CascadeClassifier(cascade)

    def get_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(gray, 1.1, 4)
        return objects


class Object:
    def __init__(self, frame, roi, object_name = "Unknown"):
        self.tracker = Tracker(frame, roi)
        self.roi = roi
        self.object_name = object_name

    def draw(self, frame):
        if self.roi is None:
            print("ROI is not known")
        else:
            cv2.rectangle(frame, (int(self.roi[0]), int(self.roi[1])), (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3])), (255, 0, 0), 2)
            cv2.putText(frame, self.object_name, (int(self.roi[0]), int(self.roi[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            return frame

    def set_roi(self, roi):
        self.roi = roi

    def init_tracker(self, frame, roi):
        self.tracker.init_tracker(frame, roi)

    def update_tracker(self, frame):
        ok, self.roi = self.tracker.update(frame)
        return ok

class Tracker:
    def __init__(self, frame, roi):
        self.tracker = cv2.TrackerKCF_create()
        self.init = self.init_tracker(frame, roi)

    def init_tracker(self, frame, roi):
        ret = self.tracker.init(frame, roi)
        return ret

    def update(self, frame):
        if self.init:
            ret, roi = self.tracker.update(frame)
            return ret, roi

class Corridor:
    def __init__(self, feed):
        self.traffic = Traffic(feed)
        self.feed = Feed(feed)
        _, self.init_frame = self.feed.read()
        self.corr_begin_p1, self.corr_begin_p2 = None, None
        self.corr_end_p1, self.corr_end_p2 = None, None

    def set_feed(self, feed):
        self.feed.set_feed(feed)

    def init_corridor(self):
        while 1:
            cv2.namedWindow('real image')
            cv2.setMouseCallback('real image', self.on_mouse, 0)
            cv2.imshow('real image', self.init_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            print('End Mouse Position: ' + str(x) + ', ' + str(y))
            end = (x, y)


    def handle_corridor(self):

        while 1:
            _, frame = self.feed.read()


            cv2.imshow('Warehouse', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

class Traffic:
    def __init__(self, feed):
        self.body_cascade = None
        self.traffic = []

    def set_body_cascade(self, cascade):
        self.body_cascade = Cascade(cascade)

    def update_bodies(self, frame):
        updated = self.get_bodies(frame)
        self.update_trackers(frame, updated)

    def update_trackers(self, frame, updated):
        for i, obj in enumerate(self.traffic):
            if i not in updated:
                obj.update_tracker(frame)

    def get_bodies(self, frame):
        #Get all bodies with fullbody cascade
        bodies = self.body_cascade.get_objects(frame)

        #List to hold all updated objects indexes
        updated = []

        #Check if bodies are found
        if bodies is not None:

            #Iterate through every body
            for body in bodies:

                #Check if body already exists
                index = self.get_object(body, 200)
                if index == -1: #Found new body
                    self.traffic.append(Object(frame, tuple(body), "Person"))
                    updated.append(len(self.traffic) - 1) #Add just added value from traffic list to updated list
                else: #Found known body
                    print("Reinit - ", body)
                    self.traffic[index].set_roi(tuple(body))
                    self.traffic[index].init_tracker(frame, tuple(body))
        return updated

    def draw_traffic(self, frame):
        for i, obj in enumerate(self.traffic):
            obj.draw(frame)

    def get_dist(self, obj1, obj2):
        #Get x, y from object 1
        x1, y1 = obj1[0], obj1[1]

        #Get x, y from object 2
        x2, y2 = obj2[0], obj2[1]

        #Get distance between objects
        ans = int(math.hypot(x2 - x1, y2 - y1))
        return ans

    def get_object(self, roi, margin):
        for i, obj in enumerate(self.traffic):
            if self.get_dist(roi, obj.roi) < margin:
                return i
        return -1

    def find_traffic(self, frame):
        self.update_bodies(frame)
        self.draw_traffic(frame)


#traffic = Traffic("img/walking.mp4")
#traffic.set_body_cascade("cascades/haarcascade_fullbody.xml")

corr = Corridor("img/walking.mp4")
corr.init_corridor()
