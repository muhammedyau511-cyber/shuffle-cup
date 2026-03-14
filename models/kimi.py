import os
import sys
import json
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from compute_accuracy import extract_answer

client = OpenAI(
    api_key="",
    base_url="https://api.moonshot.ai/v1",
)

MODEL_NAME = "kimi-k2.5"


def analyze_video(video_path, prompt_text):
    print(f"--- Reading {video_path} ---")

    with open(video_path, "rb") as f:
        video_data = f.read()

    print(f"Video loaded: {len(video_data)} bytes")

    video_url = f"data:video/{os.path.splitext(video_path)[1]};base64,{base64.b64encode(video_data).decode('utf-8')}"

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are Kimi."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ],
    )

    return completion.choices[0].message.content or ""


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

    # Videos to (re)run when not completed. None = all videos.
    target_videos = None

    # If output already exists, reload and redo N/A/target_videos or add missing target_videos
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        existing_videos = {r["video"] for r in results}
        na_indices = [
            i for i, r in enumerate(results)
            if extract_answer(r.get("response", "")) is None
            or (target_videos is not None and r.get("video") in target_videos)
        ]

        missing_target = [
            q for q in qa_data
            if (target_videos is None or q["video"] in target_videos)
            and q["video"] not in existing_videos
        ]

        if not na_indices and not missing_target:
            print("No N/A entries and no missing target videos. Nothing to redo.")
            return

        total_todo = len(na_indices) + len(missing_target)
        print(f"Found {len(na_indices)} to redo, {len(missing_target)} missing target videos ({total_todo} total).")

        count = 0
        for idx in na_indices:
            count += 1
            item = results[idx]
            video_name = item["video"]
            qa_item = next((q for q in qa_data if q["video"] == video_name), None)
            question = qa_item["question"] if qa_item else item.get("question", "")
            video_path = os.path.join(video_dir, video_name)

            print(f"\n{'='*50}")
            print(f"[{count}/{total_todo}] Redo: {video_name}")
            print(f"Old response (truncated): {item.get('response', '')[:120]}...")
            print(f"{'='*50}")

            try:
                response = analyze_video(video_path, question)
                print(f"\n--- Kimi Response ---")
                print(response)
            except Exception as e:
                response = f"Error: {e}"
                print(f"Error processing {video_name}: {e}")

            results[idx]["response"] = response
            results[idx]["model"] = MODEL_NAME

            save_results(results, output_path)
            print(f"Progress saved ({count}/{total_todo})")

        for qa_item in missing_target:
            count += 1
            video_name = qa_item["video"]
            question = qa_item["question"]
            video_path = os.path.join(video_dir, video_name)

            print(f"\n{'='*50}")
            print(f"[{count}/{total_todo}] New: {video_name}")
            print(f"Question: {question[:80]}...")
            print(f"{'='*50}")

            try:
                response = analyze_video(video_path, question)
                print(f"\n--- Kimi Response ---")
                print(response)
            except Exception as e:
                response = f"Error: {e}"
                print(f"Error processing {video_name}: {e}")

            result = qa_item.copy()
            result["model"] = MODEL_NAME
            result["response"] = response
            results.append(result)

            save_results(results, output_path)
            print(f"Progress saved ({count}/{total_todo})")

        print(f"\n{'='*50}")
        print(f"Done. Results saved to: {output_path}")
        print(f"Total processed: {count} items")
        return

    results = []

    for i, item in enumerate(qa_data):
        video_name = item["video"]
        if target_videos is not None and video_name not in target_videos:
            continue
        question = item["question"]

        video_path = os.path.join(video_dir, video_name)

        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(qa_data)}] Processing: {video_name}")
        print(f"Question: {question}")
        print(f"{'='*50}")

        try:
            response = analyze_video(video_path, question)
            print(f"\n--- Kimi Response ---")
            print(response)
            print(f"Ground Truth: {item.get('ground_truth')}")
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
