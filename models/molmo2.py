import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor, AutoModelForImageTextToText
from molmo_utils import process_vision_info

MODEL_NAME = "Molmo2-8B"
MODEL_PATH = "allenai/Molmo2-8B"
MAX_FPS = 8
MAX_NEW_TOKENS = 2048
TASKS = ["cup", "card"]
BASE_DIR = "/content/VET-Bench"

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)


def analyze_video(video_path, prompt_text):
    print(f"--- Processing {video_path} ---")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=prompt_text),
                dict(type="video", video=video_path, max_fps=MAX_FPS),
            ],
        }
    ]

    _, videos, video_kwargs = process_vision_info(messages)
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        videos=videos,
        video_metadata=video_metadatas,
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


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
            print(f"\n--- Molmo2 Response ---")
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
