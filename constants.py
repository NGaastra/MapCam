# Camera config
CAMERA_ANGLE = 70                       # Degrees
CAMERA_HEIGHT = 4                       # Meters
CAMERA_VIEWING_ANGLE = 30               # Degrees (Total)

# Feed config
FEED_WIDTH = 640                        # Pixels
FEED_HEIGHT = 480                       # Pixels

# Object search config
# General
SEARCH_MARGIN_NEW_LOC = 40              # Pixels
# Person
ROI_CROP_MARGIN = 20
# Vehicle
VEHICLE_THRESHOLD = 25                  # Intensity
MIN_VEHICLE_SIZE = 600                  # Pixels^2
TRACKER_INITIATION_DIST = 20            # Pixels

# Draw config
RECTANGLE_COLOR_PERSON = (255, 0, 0)    # BGR
RECTANGLE_COLOR_VEHICLE = (0, 255, 0)   # BGR
TEXT_COLOR = (0, 255, 255)              # BGR
