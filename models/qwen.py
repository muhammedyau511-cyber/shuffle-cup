import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import MultiModalConversation
from compute_accuracy import extract_answer

# MODEL_NAME = "qwen3-vl-flash"
# MODEL_NAME = "qwen3.5-397b-a17b"
MODEL_NAME = "qwen3-vl-8b-instruct"
# MODEL_NAME = "qwen3-vl-8b-thinking"
# MODEL_NAME = "qwen3-vl-30b-a3b-instruct"
# MODEL_NAME = "qwen3-vl-30b-a3b-thinking"
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


def analyze_video(video_path, prompt_text):
    print(f"--- Processing {video_path} ---")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # DashScope: local video via file:// (see vision doc - 传入本地文件)
    video_url = f"file://{os.path.abspath(video_path)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_url, "fps": 8},
                {"text": prompt_text},
            ],
        }
    ]

    response = MultiModalConversation.call(
        api_key=API_KEY,
        model=MODEL_NAME,
        messages=messages,
    )

    if response is None:
        raise RuntimeError("API returned no response")
    if response.output is None:
        code = getattr(response, "code", "")
        message = getattr(response, "message", "")
        parts = [p for p in (str(code), str(message)) if p]
        raise RuntimeError(f"API output is None: {' '.join(parts) or 'Unknown API error'}")

    return response.output.choices[0].message.content[0]["text"]


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
            # Recover the question from QA data (output may have stripped some keys)
            qa_item = next((q for q in qa_data if q["video"] == video_name), None)
            question = qa_item["question"] if qa_item else item.get("question", "")
            video_path = os.path.join(video_dir, video_name)

            print(f"\n{'='*50}")
            print(f"[Redo {count}/{len(na_indices)}] Processing: {video_name}")
            print(f"Old response (truncated): {item.get('response', '')[:120]}...")
            print(f"{'='*50}")

            try:
                response = analyze_video(video_path, question)
                print(f"\n--- Qwen VL Response ---")
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
            print(f"\n--- Qwen VL Response ---")
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
    print(f"Results saved to: {output_path}")
    print(f"Total processed: {len(results)} items")


if __name__ == "__main__":
    main()
