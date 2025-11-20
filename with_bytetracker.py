from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import math
import time
import os
import keyboard
import numpy as np
from collections import defaultdict


MAVLINK_CONN = "tcp:127.0.0.1:5763"  # Check out
YOLO_MODEL = "best(S6).pt"
LAST_WAYPOINT = (12.8547744, 77.4400091)  # need to place the drop point
TARGET_ALT = 8.0
SENSOR_WIDTH_MM = 5.6 / 1000.0
FOCAL_LENGTH_MM = 4.7 / 1000.0

SERVO_CHANNEL = 10          
PWM_ON = 1900               
PWM_OFF = 1100              
TRIGGER_DELAY = 3.0

object_counts = defaultdict(int)
flag = False
DROP_COOLDOWN = 20
last_drop_time = 0

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
CONFIDENCE_THRESHOLD=0.65
class TrackerArgs:
    track_thresh = 0.3
    match_thresh = 0.5
    track_buffer = 30
    mot20 = False


tracker_args = TrackerArgs()
tracker = BYTETracker(tracker_args, frame_rate=30) 

unique_object_ids = {}   # { "person": {1,3}, "pool": {5}, ... }
frame_count = 0

print("🔌 Connecting to vehicle...")
vehicle = mavutil.mavlink_connection(MAVLINK_CONN)
vehicle.wait_heartbeat()
print("✅ Connected to system (System ID:", vehicle.target_system, ")")

vehicle.mav.request_data_stream_send(
    vehicle.target_system, vehicle.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    2, 1
)

model = YOLO(YOLO_MODEL)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

os.makedirs("detections", exist_ok=True)
os.makedirs("For_Verification", exist_ok=True)

def recv_global_position_int(timeout=1):
    return vehicle.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)

def get_current_location():
    msg = recv_global_position_int(timeout=1)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0
        return lat, lon, alt
    return None

def get_current_altitude():
    cur = get_current_location()
    return cur[2] if cur else TARGET_ALT

def get_heading():
    msg = vehicle.recv_match(type="VFR_HUD", blocking=True, timeout=0.5)
    if msg and hasattr(msg, "heading"):
        return float(msg.heading)
    msg2 = recv_global_position_int(timeout=0.5)
    if msg2 and hasattr(msg2, "hdg"):
        return float(msg2.hdg) / 100.0
    return 0.0

def mode_set_and_wait(mode_name, timeout=6):
    mode_map = vehicle.mode_mapping()
    if mode_name not in mode_map:
        print(f"❌ Mode {mode_name} not supported!")
        return False
    mode_id = mode_map[mode_name]
    vehicle.mav.set_mode_send(vehicle.target_system,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              mode_id)
    start = time.time()
    while time.time() - start < timeout:
        hb = vehicle.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if not hb:
            continue
        if mavutil.mode_string_v10(hb) == mode_name:
            print(f"✅ Mode changed to {mode_name}")
            return True
    print(f"⚠ Mode change to {mode_name} timed out.")
    return False

def goto_location(lat, lon, alt):
    vehicle.mav.set_position_target_global_int_send(
        0, vehicle.target_system, vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        int(0b110111111000),
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

def get_distance_meters(loc1, loc2):
    lat1, lon1 = (loc1[0], loc1[1]) if isinstance(loc1, tuple) else (loc1.lat, loc1.lon)
    lat2, lon2 = (loc2[0], loc2[1]) if isinstance(loc2, tuple) else (loc2.lat, loc2.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.sqrt((dlat * 1.113195e5) ** 2 + (dlon * 1.113195e5) ** 2)

def get_target_location(current_location, offset_east_m, offset_north_m):
    R = 6378137.0
    dLat = offset_north_m / R
    dLon = offset_east_m / (R * math.cos(math.pi * current_location[0] / 180.0))
    newlat = current_location[0] + (dLat * 180 / math.pi)
    newlon = current_location[1] + (dLon * 180 / math.pi)
    return (newlat, newlon, current_location[2])

def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east

def compute_target_location(object_centerx, object_centery, frame_width, frame_height):
    image_centerx = frame_width // 2
    image_centery = frame_height // 2
    altitude_m = get_current_altitude()
    GSD = (SENSOR_WIDTH_MM * altitude_m) / (FOCAL_LENGTH_MM * frame_width)
    x_cam = (object_centerx - image_centerx) * GSD
    y_cam = (object_centery - image_centery) * GSD
    x_uav, y_uav = camera_to_uav(x_cam, y_cam)
    yaw_rad = math.radians(get_heading())
    north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)
    current_location = get_current_location()
    if current_location is None:
        raise RuntimeError("No GPS available to compute target location.")
    return get_target_location(current_location, east_offset, north_offset)

def has_reached_last_waypoint(vehicle_conn, last_lat, last_lon, threshold=1.0):
    msg = vehicle_conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
    if not msg:
        return False
    current_lat = msg.lat / 1e7
    current_lon = msg.lon / 1e7
    dlat = last_lat - current_lat
    dlon = last_lon - current_lon
    dist = math.sqrt((dlat * 1.113195e5) ** 2 + (dlon * 1.113195e5) ** 2)
    if dist <= threshold:
        print(f"\n🏁 Drone reached last waypoint (within {dist:.2f} m).")
        return True
    return False

def trigger_servo(channel, pwm_on_value, pwm_off_value, delay):
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm_on_value,
        0, 0, 0, 0, 0
    )
    print(f"✅ Payload servo {channel} activated (PWM {pwm_on_value})")
    time.sleep(delay)
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm_off_value,
        0, 0, 0, 0, 0
    )
    print(f"✅ Servo {channel} reset (PWM {pwm_off_value})")


def payload_drop(cap, box, frame):

    # -----------------------------
    # INITIAL TARGET FROM DETECTION
    # -----------------------------
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = frame.shape[:2]

    target_lat, target_lon, target_alt = compute_target_location(
        cx, cy, w, h
    )

    print(f"📍 Initial Target GPS: {target_lat:.7f}, {target_lon:.7f}")

    # -----------------------------
    # SWITCH TO GUIDED
    # -----------------------------
    print("🕹 Switching to GUIDED mode...")
    if not mode_set_and_wait("GUIDED", timeout=6):
        return False

    # -----------------------------
    # FLY TO INITIAL TARGET
    # -----------------------------
    print("✈️ Navigating to initial target...")
    goto_location(target_lat, target_lon, target_alt)

    while True:
        cur = get_current_location()
        if cur is None:
            time.sleep(0.2)
            continue

        dist = get_distance_meters(cur, (target_lat, target_lon, target_alt))
        print(f"Distance to initial target: {dist:.2f} m", end="\r")
        if dist < 0.8:
            print("\n✅ Initial target reached.")
            break
        time.sleep(0.4)

    # -----------------------------
    # VERIFY POOL AGAIN (RE-DETECTION)
    # -----------------------------
    print("🔍 Re-checking pool presence...")

    verify_start = time.time()
    verified = False
    corrected_lat = None
    corrected_lon = None
    pool_center = None
    
    while time.time() - verify_start < 5:
        ret, vframe = cap.read()
        if not ret:
            continue

        results = model(vframe, conf=0.5)
        found_pool = False
        pool_center = None

        if results and results[0].boxes:
            for box2 in results[0].boxes:
                cls_id = int(box2.cls[0])
                label = model.names[cls_id]
                conf = float(box2.conf[0])

                if label.upper() != "BOX":
                    continue

                bx1, by1, bx2, by2 = map(int, box2.xyxy[0])
                pool_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                found_pool = True
                break
        
        if found_pool:
            corrected_lat, corrected_lon, corrected_alt = compute_target_location(
                pool_center[0],
                pool_center[1],
                vframe.shape[1],
                vframe.shape[0],
                
            )
            print("🎯 Pool verified again!")
            verified = True
            break

        print("Pool not seen, retrying...")
        time.sleep(0.2)

    if not verified:
        print("❌ Pool was NOT verified. Aborting drop.")
        # ❗ Always restore AUTO mode
        mode_set_and_wait("AUTO", timeout=6)
        return False

    # -----------------------------
    # FLY TO CORRECTED GPS POINT
    # -----------------------------
    print("✈️ Navigating to corrected center...")
    goto_location(corrected_lat, corrected_lon, corrected_alt)

    while True:
        cur = get_current_location()
        if cur is None:
            time.sleep(0.2)
            continue

        dist = get_distance_meters(cur, (corrected_lat, corrected_lon, corrected_alt))
        print(f"Corrected distance: {dist:.2f} m", end="\r")

        if dist < 0.5:
            print("\n🎯 Corrected center reached.")
            break
        time.sleep(0.3)

    # -----------------------------
    # DROPPING PAYLOAD
    # -----------------------------
    print("💥 Dropping payload now!")
    trigger_servo(
        channel=SERVO_CHANNEL,
        pwm_on_value=PWM_ON,
        pwm_off_value=PWM_OFF,
        delay=0.8
    )

    # -----------------------------
    # ALWAYS RETURN TO AUTO
    # -----------------------------
    print("🔁 Switching back to AUTO mode...")
    if mode_set_and_wait("AUTO", timeout=6):
        print("✅ AUTO mode restored.")
    else:
        print("⚠ Failed to restore AUTO mode.")

    return True







print("🎥 Starting camera feed and object detection...")


while True:

    # ---------------------------
    # STOP CONDITIONS
    # ---------------------------
    if keyboard.is_pressed("q"):
        print("🛑 Stopping feed (keyboard interrupt).")
        break

    if has_reached_last_waypoint(vehicle, LAST_WAYPOINT[0], LAST_WAYPOINT[1]):
        print("✅ Drone reached final waypoint, stopping detection.")
        if flag:
            mode_set_and_wait("RTL", timeout=6)
        break

    # ---------------------------
    # READ CAMERA FRAME
    # ---------------------------
    ret, frame = cap.read()
    if not ret:
        print("⚠ Frame not received, retrying...")
        time.sleep(0.1)
        continue
    frame_count += 1
    annotated = frame.copy()

    # ---------------------------
    # YOLO DETECTION
    # ---------------------------
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    dets = []
    cls_ids = []
    boxes_xyxy = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])

            dets.append([x1, y1, x2, y2, conf])
            cls_ids.append(cls_id)
            boxes_xyxy.append([x1, y1, x2, y2])

    dets_np = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5))


    # ---------------------------
    # BYTE TRACK UPDATE
    # ---------------------------
    tracked_objects = tracker.update(
        dets_np,
        [frame.shape[0], frame.shape[1]],
        [frame.shape[0], frame.shape[1]]
    )


    # ---------------------------
    # MATCH TRACK → CLASS USING IoU
    # ---------------------------
    track_cls_map = {}
    track_conf_map = {} 

    def compute_iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return interArea / float(box1Area + box2Area - interArea + 1e-6)

    for track in tracked_objects:
        tx, ty, tw, th = track.tlwh
        t_box = [tx, ty, tx + tw, ty + th]

        best_iou = 0
        best_cls = -1
        best_conf = 0.0
        for idx, det_box in enumerate(boxes_xyxy):
            iou = compute_iou(t_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_cls = cls_ids[idx]
                best_conf = dets[idx][4]
        track_cls_map[track.track_id] = best_cls
        track_conf_map[track.track_id] = best_conf

    # ---------------------------
    # DRAW, SAVE, UNIQUE COUNTING
    # ---------------------------
    pool_detected = False
    pool_box = None

    for track in tracked_objects:

        # Tracker bounding box
        x, y, w, h = map(int, track.tlwh)
        x2, y2 = x + w, y + h

        cls_id = track_cls_map.get(track.track_id, -1)
        label = model.names[cls_id] if cls_id != -1 else "Unknown"
        det_conf = float(track_conf_map.get(track.track_id, 0.0))
        # ---- UNIQUE OBJECT COUNT ----
        if label not in unique_object_ids:
            unique_object_ids[label] = set()
        unique_object_ids[label].add(track.track_id)  # (never duplicates)

        # ---- DRAW ----
        cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 0), 2)
        disp_text = f"{label} ID:{track.track_id} {det_conf:.2f}"
        cv2.putText(annotated, disp_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)


        # ---- POOL DETECTION FOR PAYLOAD DROP ----
        if label.upper() == "BOX" and pool_box is None:
            pool_box = (x, y, x2, y2)
            pool_detected = True


    # ---------------------------
    # PAYLOAD DROP LOGIC
    # ---------------------------

    if frame_count % 30 == 0:
        cv2.imwrite(f"detections/frame_{int(time.time())}.jpg", annotated)
    
    if pool_detected and not flag:
        current_time = time.time()
        
        # CHECK COOLDOWN FIRST!
        if current_time - last_drop_time > DROP_COOLDOWN:
            print("🏊 Pool detected — initiating payload drop...")
            flag = True
            last_drop_time = current_time
            
            drop_success = payload_drop(cap, pool_box, frame)
            
            if drop_success:
                print("🎉 Payload drop SUCCESS")
                # flag stays True → prevents re-drops
            else:
                print("❌ Payload drop FAILED")
                flag = False  # Allow retry AFTER cooldown period
        else:
            time_remaining = DROP_COOLDOWN - (current_time - last_drop_time)
            print(f"⏳ Cooldown active: {time_remaining:.1f}s", end="\r")

# Remove the separate cooldown reset - not needed!



    # ---------------------------
    # SHOW LIVE FEED
    # ---------------------------
    cv2.imshow("Drone Camera Feed", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vehicle.close()

print("\n📊 FINAL UNIQUE OBJECT COUNTS:")
for label, ids in unique_object_ids.items():
    print(f"{label}: {len(ids)}")
