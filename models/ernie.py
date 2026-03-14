import os
import sys
import json
import base64
import mimetypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from compute_accuracy import extract_answer

client = OpenAI(
    api_key="",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",
)

MODEL_NAME = "ERNIE-4.5-VL-28B-A3B-Thinking" 

def encode_video_to_data_url(video_path):
    """Encode video to base64 data URL."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type or not mime_type.startswith("video"):
        mime_type = "video/mp4"
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        base64_video = base64.b64encode(video_bytes).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_video}"


def analyze_video(video_path, prompt_text):
    print(f"--- Processing {video_path} ---")
    
    video_data_url = encode_video_to_data_url(video_path)
    
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_data_url,
                            "fps": 4
                        }
                    }
                ]
            }
        ],
        stream=True,
        max_completion_tokens=8192
    )
    
    # Collect streaming response
    full_response = ""
    for chunk in chat_completion:
        if not chunk.choices or len(chunk.choices) == 0:
            continue
        if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
            full_response += chunk.choices[0].delta.reasoning_content
        elif chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    
    return full_response


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
                print(f"\n--- ERNIE Response ---")
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
            print(f"\n--- ERNIE Response ---")
            print(response)
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
