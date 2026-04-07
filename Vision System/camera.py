import pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os  # Added for folder management

# --- CONFIGURATION ---
FOLDER_NAME = "CSV"
CSV_FILE = os.path.join(FOLDER_NAME, "finger_data.csv")
SAVE_INTERVAL = 10 

# Create the folder if it doesn't exist
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
    print(f"Created folder: {FOLDER_NAME}")

# HSV and Camera setup
lower_yellow = np.array([20, 50, 100])
upper_yellow = np.array([35, 255, 255])
kernel = np.ones((5, 5), np.uint8)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize CSV file with headers inside the folder
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Point_ID", "X", "Y"])

last_save_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_points = []
        
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    current_points.append((i, cX, cY))
                    cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)

        # Save logic
        if time.time() - last_save_time >= SAVE_INTERVAL:
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for p_id, x, y in current_points:
                    writer.writerow([timestamp, p_id, x, y])
            
            print(f"Data saved to {CSV_FILE}")
            last_save_time = time.time()

        cv2.imshow('D405 CSV Export', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()