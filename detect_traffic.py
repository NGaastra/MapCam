import cv2


class Feed:
    def __init__(self, feed):
        self.feed = cv2.VideoCapture(feed)

    def read(self):
        return self.feed.read()

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
    def __init__(self, roi, object_name = "Unknown"):
        self.roi = roi
        self.object_name = object_name

    def draw(self, frame):
        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (255, 0, 0), 2)
        cv2.putText(frame, self.object_name, (self.roi[0], self.roi[1] - 20), "verdana", (0, 255, 255))


class Traffic:
    def __init__(self, feed):
        self.feed = Feed(feed)
        self.cascade = None

    def set_cascade(self, cascade):
        self.cascade = Cascade(cascade)

    def set_feed(self, feed):
        self.feed = Feed(feed)

    def find_traffic(self):
        while 1:
            ret, frame = self.feed.read()


            cv2.imshow('Warehouse', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

traffic = Traffic("img/warehouse2.mp4")
