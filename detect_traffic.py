import cv2
import math
import imutils
import constants
import numpy as np


# Notes
# None

# PROBLEMS ATM
# None

# Ideas
# Increase efficiency
# 1. Searching in region around new objects (maybe with background subtraction)
# 2. Don't search in ROI that already exists

# TODO
# Haar cascade for other traffic
# Make seperate frame copy for drawing, so it doesnt interfere with haar features(?)
# Add comments


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
        self.background = cv2.cvtColor(imutils.resize(cv2.imread("img/warehousebg.jpg"), width=constants.FEED_WIDTH, height=constants.FEED_HEIGHT), cv2.COLOR_BGR2GRAY)#cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    def set_background(self, background):
        self.background = background

    def get(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_fg = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        _, mask_bg = cv2.threshold(self.background, 70, 255, cv2.THRESH_BINARY_INV)
        sub = cv2.erode(mask_fg - mask_bg, None, iterations=1)
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
        self.set_roi(roi)
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
        return ok

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

    #INIT
    #Main init function (HIGHLY LIKELY TO NOT BE PERMANENT)
    def init_corridor(self):
        self.init_start_corridor()
        self.init_end_corridor()
        self.get_depth(0)
        self.draw_corridor_grid(10)
        self.draw_corridor()

    def draw_corridor_grid(self, grid_width):
        begin_length = abs(self.corr_begin_p1[0] - self.corr_begin_p2[0])
        end_length = abs(self.corr_end_p1[0] - self.corr_end_p2[0])
        block_width_begin = round(begin_length / grid_width, 0)
        block_width_end = round(end_length / grid_width, 0)
        for i in range(1, grid_width):
            begin = (int(round(self.corr_begin_p1[0] + (i * block_width_begin), 0)), self.corr_begin_p1[1])
            end = (int(round(self.corr_end_p1[0] + (i * block_width_end), 0)), self.corr_end_p1[1])
            cv2.line(self.init_frame, begin, end, (255, 0, 0), 1)


    def draw_corridor(self):
        cv2.line(self.init_frame, self.corr_begin_p1, self.corr_begin_p2, (255, 0, 0), 1)
        cv2.line(self.init_frame, self.corr_end_p1, self.corr_end_p2, (255, 0, 0), 1)
        cv2.line(self.init_frame, self.corr_begin_p1, self.corr_end_p1, (255, 0, 0), 1)
        cv2.line(self.init_frame, self.corr_begin_p2, self.corr_end_p2, (255, 0, 0), 1)
        while 1:
            cv2.imshow('Warehouse', self.init_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def init_start_corridor(self):
        cv2.namedWindow('Begin corridor')
        cv2.setMouseCallback('Begin corridor', self.start_corridor, 0)
        cv2.imshow('Begin corridor', self.init_frame)
        while 1:
            cv2.waitKey(1)
            if self.corr_begin_p1 is not None and self.corr_begin_p2 is not None:
                cv2.destroyWindow('Begin corridor')
                break

    def init_end_corridor(self):
        cv2.namedWindow('End corridor')
        cv2.setMouseCallback('End corridor', self.end_corridor, 0)
        cv2.imshow('End corridor', self.init_frame)
        while 1:
            cv2.waitKey(1)
            if self.corr_end_p1 is not None and self.corr_end_p2 is not None:
                cv2.destroyWindow('End corridor')
                break

    def start_corridor(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corr_begin_p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.corr_begin_p2 = (x, y)

    def end_corridor(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corr_end_p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.corr_end_p2 = (x, y)

    #END INIT

    def handle_corridor(self):
        while 1:
            _, frame = self.feed.read()
            self.traffic.find_traffic(frame)
            self.traffic.draw_vehicle(frame, self.traffic.get_pot_vehicles(self.feed.get_foreground(frame)))

            cv2.imshow('Warehouse', frame)
            k = cv2.waitKey(30) & 0xFF
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
                index = self.get_object(body)
                # Found new body
                if index == -1:
                    self.traffic.append(Object(frame, tuple(body), "Person"))
                    updated.append(len(self.traffic) - 1) #Add just added value from traffic list to updated list
                # Found known body
                else:
                    print("Reinit - ", body)
                    # Set new ROI
                    self.traffic[index].set_roi(tuple(body))
                    # Update tracker initialization to match new ROI
                    self.traffic[index].init_tracker(frame, tuple(body))
        return updated

    def get_pot_vehicles(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_vehicle(self, frame, contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

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


corr = Corridor("img/warehouse2.mp4")
corr.handle_corridor()
