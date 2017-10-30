import cv2
import math
import imutils
import constants
import numpy as np


# Notes
# None

# PROBLEMS ATM
# Cant convert

# Ideas
# Increase efficiency
# 1. Searching in region around new objects (maybe with background subtraction)
# 2. Don't search in ROI that already exists
# Get vehicle
# 1. Get backsub - people ROI's, search for contours in the newly created mask
# Close contours merging
# 1. Bounding box more contours (Apparently possible)

# TODO
# Add comments
# Initiate tracker when objects too close
# Delete object when vehicle is off screen

class Feed:
    def __init__(self, feed):
        self.feed = cv2.VideoCapture(feed)
        self.foreground = Foreground(self.read()[1])

    def read(self):
        ret, read = self.feed.read()
        resized = imutils.resize(read, width=constants.FEED_WIDTH, height=constants.FEED_HEIGHT)
        return ret, resized

    def set_feed(self, feed):
        self.close()
        self.feed = cv2.VideoCapture(feed)

    def set_background(self, background):
        self.foreground.set_background(background)

    def get_foreground(self, frame):
        return self.foreground.get(frame)

    def close(self):
        return self.feed.close()


class Foreground:
    def __init__(self, background):
        self.background = cv2.cvtColor(imutils.resize(cv2.imread("img/roadbg.png"), width=constants.FEED_WIDTH, height=constants.FEED_HEIGHT), cv2.COLOR_BGR2GRAY)#cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        self.subtractor = cv2.createBackgroundSubtractorMOG2()
    def set_background(self, background):
        self.background = background

    def get(self, frame):
        # Convert image to grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Kernel for morphological transformation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Get absolute difference between background and current image
        sub = cv2.absdiff(gray, self.background)

        # Add blur
        sub = cv2.bilateralFilter(sub, 9, 75, 75)

        # Threshold to convert absdiff image to binary image
        _, sub = cv2.threshold(sub, constants.VEHICLE_THRESHOLD, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

        # Fill any small holes
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel)

        # Remove noise
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        sub = cv2.dilate(sub, kernel, iterations=2)

        #detector = cv2.SimpleBlobDetector_create()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #sub = cv2.absdiff(gray, self.background)
        #sub = cv2.bilateralFilter(sub, 9, 75, 75)
        #sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, None)
        #sub = cv2.dilate(sub, None, iterations=2)
        #_, sub = cv2.threshold(sub, constants.VEHICLE_THRESHOLD, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        #keypoints = detector.detect(sub)
        #cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #sub = cv2.bitwise_not(sub)

        return sub


class Cascade:
    def __init__(self, cascade):
        self.cascade = cv2.CascadeClassifier(cascade)

    def get_objects(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Search grayscale frame for cascade match
        objects = self.cascade.detectMultiScale(gray, 1.1, 4)
        return objects

    def get_new_location(self, frame, obj):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop grayscale frame to ROI + margin size
        gray_crop = gray[int(obj.roi[1]) - constants.SEARCH_MARGIN_NEW_LOC: int(obj.roi[1] + obj.roi[3]) + constants.SEARCH_MARGIN_NEW_LOC,
                             int(obj.roi[0]) - constants.SEARCH_MARGIN_NEW_LOC: int(obj.roi[0] + obj.roi[2]) + constants.SEARCH_MARGIN_NEW_LOC]

        cv2.imshow('search', gray_crop)

        # Search in this cropped area for cascade match
        objects = self.cascade.detectMultiScale(gray_crop, 1.1, 4)
        return objects

class Object:
    def __init__(self, frame, roi, object_name = "Unknown"):
        self.tracker = Tracker(frame, roi)
        self.roi = roi
        self.bottom = (roi[0] + (roi[2] / 2), roi[1] + roi[3])
        self.object_name = object_name

    def draw(self, frame):
        # ROI is not initialized
        if self.roi is None:
            print("ROI is not known")
        # ROI is initialized
        else:
            # Draw rectangle to match ROI
            cv2.rectangle(frame, (int(self.roi[0]), int(self.roi[1])), (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3])), constants.RECTANGLE_COLOR_PERSON, 2)

            # Put object name above the object
            cv2.putText(frame, self.object_name, (int(self.roi[0]), int(self.roi[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, constants.TEXT_COLOR, 1)
            return frame

    def set_roi(self, roi):
        # Set ROI
        self.roi = roi

        # Update bottom position with new ROI
        self.bottom = (roi[0] + (roi[2] / 2), roi[1] + roi[3])

    def init_tracker(self, frame, roi):
        # Initialize tracker
        self.tracker.init_tracker(frame, roi)

    def update_tracker(self, frame):
        # Update tracker
        ok, roi = self.tracker.update(frame)

        # Set ROI to new ROI determined by the tracker
        self.set_roi(roi)
        return

class Tracker:
    def __init__(self, frame, roi):
        self.tracker = cv2.TrackerKCF_create()
        self.init = self.init_tracker(frame, roi)

    def init_tracker(self, frame, roi):
        # Initialize tracker
        ret = self.tracker.init(frame, roi)
        return ret

    def update(self, frame):
        # Tracker is initialized
        if self.init:
            # Update tracker
            ret, roi = self.tracker.update(frame)
            return ret, roi

class Corridor:
    def __init__(self, feed):
        self.traffic = Traffic(feed)
        self.traffic.set_body_cascade("cascades/haarcascade_fullbody.xml")
        self.feed = Feed(feed)
        _, self.init_frame = self.feed.read() #imutils.resize(cv2.imread("img/warehouse.jpg"), width=640, height=480)
        self.corr_begin_p1, self.corr_begin_p2 = None, None #Pass maybe as constructor argument
        self.corr_end_p1, self.corr_end_p2 = None, None

    def set_feed(self, feed):
        # Set feed
        self.feed.set_feed(feed)

    def map(self, x, in_min, in_max, out_min, out_max):
        # Map range of values to new range of values
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_depth(self, y):
        # Calculate angle that corresponds with the current position of the object
        angle = self.map(y, 0, constants.FEED_HEIGHT, (constants.CAMERA_ANGLE + (constants.CAMERA_VIEWING_ANGLE / 2)), (constants.CAMERA_ANGLE - (constants.CAMERA_VIEWING_ANGLE / 2)))

        # Calculate distance based on camera height and camera angle
        dist = constants.CAMERA_HEIGHT * math.tan(math.radians(angle))
        return dist

    def get_position(self, obj):
        pos = (obj.bottom[0], self.get_depth(obj.bottom[1]))
        return pos

    def handle_corridor(self):
        while 1:
            _, frame = self.feed.read()
            fg = self.feed.get_foreground(frame)
            self.traffic.find_traffic(frame)
            self.traffic.get_pot_vehicles(frame, fg)
            #self.traffic.draw_vehicle(frame, self.traffic.get_pot_vehicles(fg))

            cv2.imshow('Warehouse', frame)
            cv2.imshow('FG', fg)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break

class Traffic:
    def __init__(self, feed):
        self.body_cascade = None
        self.traffic = []

    def set_body_cascade(self, cascade):
        self.body_cascade = Cascade(cascade)

    def update_trackers(self, frame, updated):
        for i, obj in enumerate(self.traffic):
            if i not in updated:
                obj.update_tracker(frame)

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

    def get_object(self, roi):
        # Iterate through every object in traffic
        for i, obj in enumerate(self.traffic):
            # If distance is smaller than SEARCH_MARGIN_NEW_LOC
            if self.get_dist(roi, obj.roi) < constants.SEARCH_MARGIN_NEW_LOC:
                # Return index
                return i
        return -1

    def find_traffic(self, frame):
        self.update_bodies(frame)
        self.draw_traffic(frame)


corr = Corridor("img/road.mp4")
corr.handle_corridor()
