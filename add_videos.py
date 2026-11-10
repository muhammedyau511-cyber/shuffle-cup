import json
import os

# --- CHANGE THESE FOR YOUR VIDEO ---
VIDEO_NAME = "my_video.mp4"
FINAL_POSITION = "Left"  # Options: "Left", "Middle", or "Right"
# -----------------------------------

ANSWER_MAP = {"Left": "A", "Middle": "B", "Right": "C"}
dir_path = "vetbench/cup"

# 1. Update cup.json
cup_path = os.path.join(dir_path, "cup.json")
with open(cup_path, "r") as f:
    cup_data = json.load(f)

# Remove if already exists, then add new generic config
cup_data = [x for x in cup_data if x.get("video") != VIDEO_NAME]
cup_data.append({
    "video": VIDEO_NAME,
    "fps": 30,
    "total_frames": 300,
    "resolution": [640, 480],
    "task": "cup",
    "game_settings": {"num_objects": 3, "num_swaps": 5},
    "ground_truth": FINAL_POSITION,
    "initial": {"ball": FINAL_POSITION}
})
with open(cup_path, "w") as f:
    json.dump(cup_data, f, indent=2)

# 2. Update QA.json
qa_path = os.path.join(dir_path, "QA.json")
with open(qa_path, "r") as f:
    qa_data = json.load(f)

qa_data = [x for x in qa_data if x.get("video") != VIDEO_NAME]
qa_data.append({
    "video": VIDEO_NAME,
    "question": "Which cup contains the ball at the end of the video? (A) Left (B) Middle (C) Right",
    "answer": ANSWER_MAP[FINAL_POSITION]
})
with open(qa_path, "w") as f:
    json.dump(qa_data, f, indent=2)

print(f"Successfully configured {VIDEO_NAME} with final position: {FINAL_POSITION}!")

