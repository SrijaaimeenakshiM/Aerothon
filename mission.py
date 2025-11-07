from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
import time
import math
import cv2
from ultralytics import YOLO
from collections import deque
from dronekit import LocationGlobalRelative

vehicle = connect("tcp:127.0.0.1:5763", wait_ready=True)
print("✅ Connected to vehicle")


def load_mission(filename):
    cmds = vehicle.commands
    cmds.clear()

    waypoint_queue = deque()  # Initialize queue

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("QGC") or line.strip() == "":
                continue
            parts = line.split("\t")
            if len(parts) > 10:
                try:
                    lat = float(parts[8])
                    lon = float(parts[9])
                    alt = float(parts[10])
                    cmd = Command(0, 0, 0, int(parts[2]), int(parts[3]),
                                  0, 0, float(parts[4]), float(parts[5]),
                                  float(parts[6]), float(parts[7]),
                                  lat, lon, alt)
                    cmds.add(cmd)
                    waypoint_queue.append((lat, lon, alt))  # Add to queue here
                except:
                    continue

    cmds.upload()
    print("✅ Mission uploaded successfully!")
    return waypoint_queue  


def arm_and_takeoff(aTargetAltitude):
    print("🚀 Arming motors...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)
    print("Taking off...")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")
        if alt >= aTargetAltitude * 0.95:
            print(f"✅ Reached target altitude ({alt:.2f} m)")
            break
        time.sleep(1)
    print("Switching to AUTO mode to start mission...")
    vehicle.mode = VehicleMode("AUTO")
    while vehicle.mode.name != "AUTO":
        print(" Waiting for AUTO mode...")
        time.sleep(1)
    print("✅ AUTO mission started!")

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


sensor_width_mm = 7.6
focal_length_mm = 4.4
altitude_m = 15
model = YOLO("best_model_in_the_world.pt")

def detect_and_navigate(frame):
    results = model(frame)
    if results and results[0].boxes:
        annotated = results[0].plot()
        cv2.imwrite("detected_obj.jpg", annotated)
        box = results[0].boxes[0]
        xmin, ymin, xmax, ymax = box.xyxy[0]
        object_centerx = int((xmin + xmax) / 2)
        object_centery = int((ymin + ymax) / 2)
        return object_centerx, object_centery
    return None

def compute_target_location(object_centerx, object_centery, frame_width, frame_height):
    image_centerx = frame_width // 2
    image_centery = frame_height // 2

    GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * frame_width)

    x_cam = (object_centerx - image_centerx) * GSD
    y_cam = (object_centery - image_centery) * GSD

    x_uav, y_uav = camera_to_uav(x_cam, y_cam)
    yaw_rad = math.radians(vehicle.heading)
    north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)

    current_location = vehicle.location.global_relative_frame
    target_location = get_target_location(current_location, east_offset, north_offset)
    return target_location


def monitor_mission_with_image(waypoint_queue, target_image_path, trigger_delay=10):
   
    last_wp = -1
    waypoint_queue.popleft()
    # Load the target image once
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        print("❌ Error: Target image not found.")
        return

    # Wait a few seconds after starting survey before triggering detection
    print(f"⏳ Waiting {trigger_delay} seconds before detecting target...")
    time.sleep(trigger_delay)

    # Run detection once using the target image
    detection = detect_and_navigate(target_img)
    if detection:
        print("🎯 Object detected! Switching to GUIDED mode...")
        vehicle.mode = VehicleMode("GUIDED")

        (cx, cy) = detection
        h, w = target_img.shape[:2]
        target_location = compute_target_location(cx, cy, w, h)
        print(f"Target GPS: {target_location.lat:.7f}, {target_location.lon:.7f}")

        # Fly to detected object
        vehicle.simple_goto(target_location)
        while True:
            dist = get_distance_meters(vehicle.location.global_relative_frame, target_location)
            if dist < 0.8:
                print("✅ Object reached.")
                break
            time.sleep(1)

        # Resume AUTO from next waypoint in the queue
        if waypoint_queue:
            lat, lon, alt = waypoint_queue[0]
            print(f"Resuming AUTO → next waypoint: {lat}, {lon}, {alt}")
            vehicle.mode = VehicleMode("GUIDED")
            vehicle.simple_goto(LocationGlobalRelative(lat, lon, alt))
            while True:
                target_location = LocationGlobalRelative(lat, lon, vehicle.location.global_relative_frame.alt)
                dist = get_distance_meters(vehicle.location.global_relative_frame, target_location)
                if dist < 1.0:
                    print("✅ Reached next waypoint from queue, resuming AUTO.")
                    break
                time.sleep(1)
            vehicle.mode = VehicleMode("AUTO")
            while vehicle.mode.name != "AUTO":
                time.sleep(0.5)
            time.sleep(1)

    # Normal waypoint monitoring continues after the detour
    while True:
        next_wp = vehicle.commands.next
        mode = vehicle.mode.name

        if next_wp != last_wp:
            if last_wp >= 0 and waypoint_queue:
                finished = waypoint_queue.popleft()
                print(f"✅ Waypoint reached: {finished}")
                print(f"Remaining queue: {list(waypoint_queue)}")
            last_wp = next_wp

        if not waypoint_queue:
            print("🏁 Mission completed. Landing...")
            vehicle.mode = VehicleMode("LAND")
            break

        time.sleep(1)




mission_file = "wp3.waypoints"
print("📘 Loading mission...")
waypoint_queue = load_mission(mission_file)
print(f"\nLoaded {len(waypoint_queue)} waypoints")

arm_and_takeoff(altitude_m)
monitor_mission_with_image(waypoint_queue,"test_pic2 (1).jpg")

vehicle.close()
print("✅ Mission and detection complete.")
