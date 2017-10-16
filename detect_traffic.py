import cv2
import math
import numpy as np
import imutils


#PROBLEMS ATM
#Region of interest not being updated when calling tracker!!! (Tracker problem? Problem in code? POINTER PROBLEM!?)
#Region of interest is updated while known object is found and then changed back to original position when tracker is updated


class Feed:
    def __init__(self, feed):
        self.feed = cv2.VideoCapture(feed)

    def read(self):
        ret, read = self.feed.read()
        crop = imutils.resize(read, width=640, height=480)
        return ret, crop

    def close(self):
        return self.feed.close()


class Cascade:
    def __init__(self, cascade):
        self.cascade = cv2.CascadeClassifier(cascade)

    def get_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(gray, 1.1, 2)
        return objects


class Object:
    def __init__(self, frame, roi, object_name = "Unknown"):
        self.tracker = Tracker(frame, roi)
        self.roi = roi
        self.object_name = object_name
        self.frame = frame

    def draw(self, frame):
        cv2.rectangle(frame, (int(self.roi[0]), int(self.roi[1])), (int(self.roi[0] + self.roi[2]), int(self.roi[1]+ self.roi[3])), (255, 0, 0), 2)
        cv2.putText(frame, self.object_name, (int(self.roi[0]), int(self.roi[1]) - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    def set_roi(self, roi):
        self.roi = roi

    def update_tracker(self):
        ok, self.roi = self.tracker.update(self.frame)
        return ok

class Tracker:
    def __init__(self, frame, roi):
        self.init, self.tracker = self.init_tracker(frame, roi)

    def init_tracker(self, frame, roi):
        tracker = cv2.TrackerKCF_create()
        ret = tracker.init(frame, tuple(roi))
        return ret, tracker

    def update(self, frame):
        ret, roi = self.tracker.update(frame)
        return ret, roi


class Traffic:
    def __init__(self, feed):
        self.feed = Feed(feed)
        self.body_cascade = None
        self.traffic = []

    def set_body_cascade(self, cascade):
        self.body_cascade = Cascade(cascade)

    def update_bodies(self, frame):
        updated = self.get_bodies(frame)
        #self.update_trackers(updated)

    def update_trackers(self, updated):
        for i, obj in enumerate(self.traffic):
            if i not in updated:
                obj.update_tracker()

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
                index = self.get_object(body, 70)
                if index == -1: #Found new body
                    self.traffic.append(Object(frame, tuple(body), "Person"))
                    updated.append(len(self.traffic) - 1) #Add just added value from traffic list to updated list
                else: #Found known body
                    self.traffic[index].set_roi(tuple(body))
        return updated

    def draw_traffic(self, frame):
        for i, obj in enumerate(self.traffic):
            obj.draw(frame)

    def set_feed(self, feed):
        self.feed = Feed(feed)

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

    def find_traffic(self):
        while 1:
            ret, frame = self.feed.read()
            self.update_bodies(frame)
            self.draw_traffic(frame)

            cv2.imshow('Warehouse', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

traffic = Traffic("img/walking4.mp4")
traffic.set_body_cascade("cascades/haarcascade_fullbody.xml")

traffic.find_traffic()
