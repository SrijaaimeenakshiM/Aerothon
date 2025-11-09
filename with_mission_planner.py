from dronekit import connect, VehicleMode, LocationGlobalRelative
from ultralytics import YOLO
import cv2
import math
import time
import os
import keyboard   


print("🔌 Connecting to vehicle...")
vehicle = connect("tcp:127.0.0.1:5763", wait_ready=True)
print("✅ Connected to vehicle")


model = YOLO("best_model_in_the_world.pt")


print("🕓 Waiting for mission takeoff to start...")
while vehicle.commands.next == 0:
    time.sleep(1)

print("🚀 Takeoff initiated...")

while vehicle.commands.next == 1:
    alt = vehicle.location.global_relative_frame.alt
    print(f"   Current Altitude: {alt:.2f} m")
    time.sleep(1)

print("✅ Takeoff complete! Starting camera...")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

os.makedirs("detections", exist_ok=True)

sensor_width_mm = 7.6
focal_length_mm = 4.4



def get_target_location(current_location, offset_east_m, offset_north_m):
    R = 6378137.0
    dLat = offset_north_m / R
    dLon = offset_east_m / (R * math.cos(math.pi * current_location.lat / 180.0))
    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, current_location.alt)


def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam


def uav_to_ne(x_uav, y_uav, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east


def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat * 1.113195e5) ** 2 + (dlon * 1.113195e5) ** 2)


def compute_target_location(object_centerx, object_centery, frame_width, frame_height):
    image_centerx = frame_width // 2
    image_centery = frame_height // 2
    altitude_m = vehicle.location.global_relative_frame.alt
    GSD = (sensor_width_mm * altitude_m) / (focal_length_mm * frame_width)

    x_cam = (object_centerx - image_centerx) * GSD
    y_cam = (object_centery - image_centery) * GSD

    x_uav, y_uav = camera_to_uav(x_cam, y_cam)
    yaw_rad = math.radians(vehicle.heading)
    north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)

    current_location = vehicle.location.global_relative_frame
    return get_target_location(current_location, east_offset, north_offset)


print("🎥 Starting camera feed and object detection...")

while True:
    
    if keyboard.is_pressed("q"):
        print("🛑 Stopping feed (keyboard interrupt).")
        break

    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received, retrying...")
        continue

    results = model(frame)
    annotated = frame.copy()

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "TARGET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            filename = f"detections/detect_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"🎯 Object detected → {filename}")

            # Switch to GUIDED
            print("🕹 Switching to GUIDED mode...")
            vehicle.mode = VehicleMode("GUIDED")
            while vehicle.mode.name != "GUIDED":
                
                time.sleep(0.5)

            # Compute and go to target
            h, w = frame.shape[:2]
            target_location = compute_target_location(cx, cy, w, h)
            print(f"📍 Target GPS: {target_location.lat:.7f}, {target_location.lon:.7f}")
            vehicle.simple_goto(target_location)

            # Wait till reach
            while True:
                dist = get_distance_meters(vehicle.location.global_relative_frame, target_location)
                print(f"Distance to target: {dist:.2f} m", end="\r")
                if dist < 0.8:
                    print("\n✅ Object reached.")
                    break
               
                time.sleep(0.5)

            # Resume AUTO
            print("🔁 Switching back to AUTO mode...")
            vehicle.mode = VehicleMode("AUTO")
            while vehicle.mode.name != "AUTO":
                
                time.sleep(0.5)
            print("✅ Resumed AUTO mission.\n")
            time.sleep(2)

    cv2.imshow("Drone Camera Feed", annotated)

    # If mission completed
    mode = vehicle.mode.name
    if mode in ["RTL", "LAND"]:
        print("🛬 Mission completed, stopping camera.")
        break

cap.release()
cv2.destroyAllWindows()
vehicle.close()
print("✅ Detection session ended.")
