from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import cv2
from ultralytics import YOLO  
from pymavlink import mavutil

# Connect to the drone
vehicle = connect("tcp:127.0.0.1:5763", wait_ready=True)

# Camera specs (mm and pixels — tune these for your camera)
sensor_width_mm = 7.6      
focal_length_mm = 4.4      
altitude_m = 10            # Target altitude

# Arm and takeoff
def arm_and_takeoff(aTargetAltitude):
    print("Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(5)
    print("Taking off...")
    vehicle.simple_takeoff(aTargetAltitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= aTargetAltitude * 0.95:
            print("Reached target altitude.")
            break
        time.sleep(1)

# Pixel offset (East, North) to GPS
def get_target_location(current_location, offset_east_m, offset_north_m):
    R = 6378137.0
    dLat = offset_north_m / R
    dLon = offset_east_m / (R * math.cos(math.pi * current_location.lat / 180.0))

    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, current_location.alt)

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat * 1.113195e5)**2 + (dlon * 1.113195e5)**2)


def camera_to_uav(x_cam, y_cam):
    x_uav = -y_cam  
    y_uav = x_cam   
    return x_uav, y_uav

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east

# Load YOLO model
model = YOLO('S:/drone/best_model_in_the_world.pt')

# Start mission
arm_and_takeoff(altitude_m)

# Read image for simulation
frame = cv2.imread("test_pic3.jpg")
image_width_px = frame.shape[1]
image_height_px = frame.shape[0]

# GSD: meters per pixel
GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * image_width_px)

results = model(frame)
if results and results[0].boxes:
    annotated_frame = results[0].plot()
    cv2.imwrite("detected_1.jpg", annotated_frame)

    box = results[0].boxes[0]
    xmin, ymin, xmax, ymax = box.xyxy[0]
    object_centerx = int((xmin + xmax) / 2)
    object_centery = int((ymin + ymax) / 2)

    image_centerx = image_width_px // 2
    image_centery = image_height_px // 2

    # Pixel offset in meters (camera frame)
    x_cam = (object_centerx - image_centerx) * GSD
    y_cam = (object_centery - image_centery) * GSD

    # Convert to UAV frame
    x_uav, y_uav = camera_to_uav(x_cam, y_cam)

    # Convert to North-East using current heading
    yaw_rad = math.radians(vehicle.heading)
    north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)

    print(f"Offsets (m): Camera: ({x_cam:.2f}, {y_cam:.2f}) → UAV: ({x_uav:.2f}, {y_uav:.2f}) → NE: ({north_offset:.2f}, {east_offset:.2f})")

    # Get new GPS target
    current_location = vehicle.location.global_relative_frame
    target_location = get_target_location(current_location, east_offset, north_offset)

    print(f"Target Location: lat={target_location.lat}, lon={target_location.lon}")
    print("Navigating to target...")
    vehicle.simple_goto(target_location)

    while True:
        current_loc = vehicle.location.global_relative_frame
        distance = get_distance_meters(current_loc, target_location)
        if distance < 0.8:
            print("Target reached.")
            break
        time.sleep(1)

# Land
print("Landing...")
vehicle.mode = VehicleMode("LAND")
vehicle.close()
