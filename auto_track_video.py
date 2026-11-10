import json
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_NAME = "my_video.mp4"  # Change this to your video filename
DIR_PATH = "vetbench/cup"
# --------------------

video_path = os.path.join(DIR_PATH, VIDEO_NAME)
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Could not find video at {video_path}. Please place it there first.")

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Analyzing {VIDEO_NAME} ({total_frames} frames)... Please wait.")

# We will look at the final frames of the video to see where the motion or object settles
# Grab the last 10 frames to determine final position
last_positions = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # If we are near the end of the video, track horizontal position
    if frame_count >= total_frames - int(fps * 1.5): # Last 1.5 seconds
        # Convert to HSV to easily detect colored objects (adjust if needed, or track high intensity)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Example color mask (Default looks for bright/saturated colors like a distinct ball). 
        # If your ball is red/orange, this range works well:
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find coordinates of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 20: # Filter out tiny noise
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    last_positions.append(cx)

cap.release()

# Determine final position based on screen width division (Left, Middle, Right thirds)
if last_positions:
    avg_x = sum(last_positions) / len(last_positions)
    third = width / 3
    if avg_x < third:
        final_position = "Left"
    elif avg_x < (third * 2):
        final_position = "Middle"
    else:
        final_position = "Right"
else:
    # Fallback default if color threshold didn't catch a distinct standalone color
    print("Warning: Could not isolate a distinct colored ball automatically. Defaulting to 'Left'. You can adjust color bounds in the script if needed.")
    final_position = "Left"

answer_map = {"Left": "A", "Middle": "B", "Right": "C"}

# 1. Update cup.json
cup_path = os.path.join(DIR_PATH, "cup.json")
with open(cup_path, "r") as f:
    cup_data = json.load(f)

cup_data = [x for x in cup_data if x.get("video") != VIDEO_NAME]
cup_data.append({
    "video": VIDEO_NAME,
    "fps": fps if fps > 0 else 30,
    "total_frames": total_frames,
    "resolution": [width, height],
    "task": "cup",
    "game_settings": {"num_objects": 3, "num_swaps": 5},
    "ground_truth": final_position,
    "initial": {"ball": final_position}
})
with open(cup_path, "w") as f:
    json.dump(cup_data, f, indent=2)

# 2. Update QA.json
qa_path = os.path.join(DIR_PATH, "QA.json")
with open(qa_path, "r") as f:
    qa_data = json.load(f)

qa_data = [x for x in qa_data if x.get("video") != VIDEO_NAME]
qa_data.append({
    "video": VIDEO_NAME,
    "question": "Which cup contains the ball at the end of the video? (A) Left (B) Middle (C) Right",
    "answer": answer_map[final_position]
})
with open(qa_path, "w") as f:
    json.dump(qa_data, f, indent=2)

print(f"Success! Detected final position as: {final_position}")
print("Automatically updated cup.json and QA.json!")
