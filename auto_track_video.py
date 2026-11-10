import json
import os
import cv2
import numpy as np

DIR_PATH = "vetbench/cup"
cup_path = os.path.join(DIR_PATH, "cup.json")
qa_path = os.path.join(DIR_PATH, "QA.json")

# Find all mp4 videos in the folder
video_files = [f for f in os.listdir(DIR_PATH) if f.endswith(".mp4")]

if not video_files:
    print(f"No .mp4 videos found in {DIR_PATH}.")
    exit()

# Load existing JSON data
with open(cup_path, "r") as f:
    cup_data = json.load(f)

with open(qa_path, "r") as f:
    qa_data = json.load(f)

answer_map = {"Left": "A", "Middle": "B", "Right": "C"}

for video_name in video_files:
    video_path = os.path.join(DIR_PATH, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        continue
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nProcessing: {video_name} ({total_frames} frames)...")

    last_positions = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track near the end of the video (last 1.5 seconds)
        if frame_count >= total_frames - int(max(fps, 1) * 1.5):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 100, 100])
            upper_bound = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 20:
                    M = cv2.moments(largest_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        last_positions.append(cx)
        frame_count += 1

    cap.release()

    # Determine final position based on screen thirds
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
        final_position = "Left"  # Fallback

    print(f" -> Detected final position: {final_position}")

    # Update cup.json entry
    cup_data = [x for x in cup_data if x.get("video") != video_name]
    cup_data.append({
        "video": video_name,
        "fps": fps if fps > 0 else 30,
        "total_frames": total_frames,
        "resolution": [width, height],
        "task": "cup",
        "game_settings": {"num_objects": 3, "num_swaps": 5},
        "ground_truth": final_position,
        "initial": {"ball": final_position}
    })

    # Update QA.json entry
    qa_data = [x for x in qa_data if x.get("video") != video_name]
    qa_data.append({
        "video": video_name,
        "question": "Which cup contains the ball at the end of the video? (A) Left (B) Middle (C) Right",
        "answer": answer_map[final_position]
    })

# Save updated JSONs back out
with open(cup_path, "w") as f:
    json.dump(cup_data, f, indent=2)

with open(qa_path, "w") as f:
    json.dump(qa_data, f, indent=2)

print("\nAll videos processed successfully! cup.json and QA.json updated automatically.")
