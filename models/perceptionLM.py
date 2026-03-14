# !pip install -q transformers accelerate pillow torch av qwen-vl-utils decord ffmpeg
import os
import json
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_NAME = "Perception-LM-8B"
MODEL_PATH = "facebook/Perception-LM-8B"
NUM_FRAMES = 32
MAX_NEW_TOKENS = 256
TASKS = ["cup", "card"]
BASE_DIR = "/content/VET-Bench"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to("cuda")


def analyze_video(video_path, prompt_text):
    print(f"--- Processing {video_path} ---")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        [conversation],
        num_frames=NUM_FRAMES,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_load_backend="decord",
    )
    inputs = inputs.to(model.device, model.dtype)

    generate_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    input_length = inputs["input_ids"].shape[1]
    output_ids = generate_ids[:, input_length:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def save_results(results, output_path):
    keys_to_drop = {"task", "ground_truth"}
    to_save = [{k: v for k, v in r.items() if k not in keys_to_drop} for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)


for task in TASKS:
    qa_path = os.path.join(BASE_DIR, task, "QA.json")
    video_dir = os.path.join(BASE_DIR, task)
    output_path = os.path.join(BASE_DIR, "output", task, f"{MODEL_NAME}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []

    for i, item in enumerate(qa_data):
        video_name = item["video"]
        question = item["question"]
        video_path = os.path.join(video_dir, video_name)

        print(f"\n{'='*50}")
        print(f"[{task}] [{i+1}/{len(qa_data)}] Processing: {video_name}")
        print(f"Question: {question}")
        print(f"{'='*50}")

        try:
            response = analyze_video(video_path, question)
            print(f"\n--- PerceptionLM Response ---")
            print(response)
            print(f"Ground Truth Answer: {item.get('answer')}")
        except Exception as e:
            response = f"Error: {e}"
            print(f"Error processing {video_name}: {e}")

        result = item.copy()
        result["model"] = MODEL_NAME
        result["response"] = response
        results.append(result)

        save_results(results, output_path)
        print(f"Progress saved ({len(results)} items)")

    print(f"\n{'='*50}")
    print(f"[{task}] Results saved to: {output_path}")
    print(f"[{task}] Total processed: {len(results)} items")
