import cv2


class Feed:
    def __init__(self, feed):
        self.feed = cv2.VideoCapture(feed)

    def read(self):
        return self.feed.read()


class Cascade:
    def __init__(self, cascade):
        self.cascade = cv2.CascadeClassifier(cascade)

    def get_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(gray, 1.3, 5)
        return objects


class Object:
    def __init__(self, object_name, roi):
        self.object_name = object_name
        self.roi = roi

    def draw(self, frame):
        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (255, 0, 0), 2)
        cv2.putText(frame, self.object_name, (self.roi[0], self.roi[1] - 20), "verdana", (0, 255, 255))
