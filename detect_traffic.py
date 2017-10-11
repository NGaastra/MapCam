import cv2
import math
import numpy as np
import imutils


#PROBLEMS ATM
#Region of interest not being updated when calling tracker
#Maybe something wrong with drawing staying on screen (Probably from problem 1)



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
        objects = self.cascade.detectMultiScale(gray, 1.3, 5)
        return objects


class Object:
    def __init__(self, frame, roi, object_name = "Unknown"):
        self.tracker = Tracker(frame, roi)
        self.roi = roi
        self.p1, self.p2 = (roi[0], roi[1]), (roi[0] + roi[2], roi[1]+ roi[3])
        self.object_name = object_name
        self.frame = frame

    def draw(self, frame):
        cv2.rectangle(frame, self.p1, self.p2, (255, 0, 0), 2)
        cv2.putText(frame, self.object_name, (self.p1[0], self.p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    def set_roi(self, roi):
        self.roi = roi

    def update_tracker(self):
        print(self.roi)
        ok, self.roi = self.tracker.update(self.frame)
        return ok

class Tracker:
    def __init__(self, frame, roi):
        self.init, self.tracker = self.init_tracker(frame, roi)

    def init_tracker(self, frame, roi):
        tracker = cv2.TrackerMedianFlow_create()
        ret = tracker.init(frame, tuple(roi))
        return ret, tracker

    def update(self, frame):
        ret, roi = self.tracker.update(frame)
        print(ret, " - ", roi, "\n")
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
        self.update_trackers(updated)

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
                index = self.get_object(body, 5)
                #print(body)
                if index == -1: #Found new body
                    self.traffic.append(Object(frame, body, "Person"))
                    updated.append(len(self.traffic) - 1) #Add just added value from traffic list to updated list
                else: #Found known body
                    #print("Index ", index, " Old: ", self.traffic[index].roi, "  New: ", body)
                    self.traffic[index].roi = body
                    #print(self.traffic[index].roi)
        return updated

    def draw_traffic(self, frame):
        for obj in self.traffic:
            obj.draw(frame)

    def set_feed(self, feed):
        self.feed = Feed(feed)

    def get_dist(self, object1, object2):
        x1, y1 = object1[0], object1[1]
        x2, y2 = object2[0], object2[1]
        ans = int(math.hypot(x2 - x1, y2 - y1))
        return ans

    def get_object(self, roi, margin):
        for i, obj2 in enumerate(self.traffic):
            if self.get_dist(roi, obj2.roi) < margin:
                return i
        return -1

    def find_traffic(self):
        obj = []
        person = None
        while 1:
            ret, frame = self.feed.read()
            #self.update_bodies(frame)
            #self.draw_traffic(frame)
            if len(obj) is 0:
                obj = self.body_cascade.get_objects(frame)
                for obj1 in obj:
                    person = Object(frame, obj1)

            if len(obj) is not 0:
                person.update_tracker()
            cv2.imshow('Warehouse', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

traffic = Traffic("img/walking1.mp4")
traffic.set_body_cascade("cascades/haarcascade_fullbody.xml")

traffic.find_traffic()