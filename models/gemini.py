import time
import os
import sys
import json
import base64

# Add parent dir so compute_accuracy can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from compute_accuracy import extract_answer


client = genai.Client(api_key="")

MODEL_NAME = "gemini-3-flash-preview"
# MODEL_NAME = "gemini-2.5-pro"
# MODEL_NAME = "gemini-3-pro-preview"

def analyze_video(video_path, prompt_text):
    print(f"--- Reading {video_path} ---")
    
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    print(f"Video loaded: {len(video_data)} bytes")
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="video/mp4",
                    data=video_data,
                ),
                types.Part.from_text(text=prompt_text),
            ],
        ),
    ]

    # 4. Generate Content with Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=1,
        )
    )

    return response.text


def save_results(results, output_path):
    """Save results to file - called after each item to preserve progress."""
    keys_to_drop = {"task", "ground_truth"}
    to_save = [{k: v for k, v in r.items() if k not in keys_to_drop} for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Video directory (contains videos + QA.json)
    video_dir = os.path.join(script_dir, "../vetbench/cup")
    qa_path = os.path.join(video_dir, "QA.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    output_path = os.path.join(script_dir, "..", "output", "cup", f"{MODEL_NAME}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # If output already exists, reload and only redo N/A entries
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        na_indices = [
            i for i, r in enumerate(results)
            if extract_answer(r.get("response", "")) is None
        ]

        if not na_indices:
            print("No N/A entries found. Nothing to redo.")
            return

        print(f"Found {len(na_indices)} N/A entries to redo.")

        for count, idx in enumerate(na_indices, 1):
            item = results[idx]
            video_name = item["video"]
            qa_item = next((q for q in qa_data if q["video"] == video_name), None)
            question = qa_item["question"] if qa_item else item.get("question", "")
            video_path = os.path.join(video_dir, video_name)

            print(f"\n{'='*50}")
            print(f"[Redo {count}/{len(na_indices)}] Processing: {video_name}")
            print(f"Old response (truncated): {item.get('response', '')[:120]}...")
            print(f"{'='*50}")

            try:
                response = analyze_video(video_path, question)
                print(f"\n--- Gemini Response ---")
                print(response)
            except Exception as e:
                response = f"Error: {e}"
                print(f"Error processing {video_name}: {e}")

            results[idx]["response"] = response
            results[idx]["model"] = MODEL_NAME

            save_results(results, output_path)
            print(f"Progress saved (redone {count}/{len(na_indices)})")

        print(f"\n{'='*50}")
        print(f"Redo complete. Results saved to: {output_path}")
        print(f"Total redone: {len(na_indices)} items")
        return

    results = []
    
    for i, item in enumerate(qa_data):
        video_name = item["video"]
        question = item["question"]

        video_path = os.path.join(video_dir, video_name)

        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(qa_data)}] Processing: {video_name}")
        print(f"Question: {question}")
        print(f"{'='*50}")

        try:
            response = analyze_video(video_path, question)
            print(f"\n--- Gemini Response ---")
            print(response)
            print(f"Ground Truth: {item.get('ground_truth')}")
            print(f"Ground Truth Answer: {item.get('answer')}")
        except Exception as e:
            response = f"Error: {e}"
            print(f"Error processing {video_name}: {e}")
        
        # Copy all fields from original item and add model/response
        result = item.copy()
        result["model"] = MODEL_NAME
        result["response"] = response
        results.append(result)
        
        # Save after each item to preserve progress
        save_results(results, output_path)
        print(f"Progress saved ({len(results)} items)")
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print(f"Total processed: {len(results)} items")


if __name__ == "__main__":
    main()

