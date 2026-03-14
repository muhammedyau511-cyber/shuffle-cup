#!/usr/bin/env python3
"""
Run Gemini 3 inference on cup_games_metadata_filtered.json and compute accuracy.
Videos are stored in the cup_games/ folder.
"""

import time
import os
import json
import re
from google import genai
from google.genai import types


# 1. Setup Client
client = genai.Client(api_key="")

MODEL_NAME = "gemini-3-pro-preview"

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def analyze_video(video_path, prompt_text):
    """Send a video + prompt to Gemini and return the response text."""
    with open(video_path, "rb") as f:
        video_data = f.read()

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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )

    return response.text


def build_prompt(question, options):
    """Build a multiple-choice prompt from the question and options list."""
    prompt = question + "\n\n"
    for i, opt in enumerate(options):
        letter = OPTION_LETTERS[i]
        prompt += f"{letter}. {opt}\n"
    prompt += (
        "\nPlease select the correct answer from the options above. "
        "Put your final answer in \\boxed{X} where X is the letter of your choice."
    )
    return prompt


def extract_answer(response: str) -> str | None:
    """Extract the answer letter from model response, looking for \\boxed{X} pattern."""
    patterns = [
        r'\$\\boxed\{([A-Za-z])\}\$',
        r'\\boxed\{([A-Za-z])\}',
        r'\$\\boxed\{([A-Za-z])\s*\}\$',
        r'\$\\boxed\{([A-Za-z])[.\s].*?\}\$',
        r'\\boxed\{([A-Za-z])[.\s].*?\}',
        r'\$\\boxed\{\\text\{([A-Za-z])\}\}\$',
        r'\\boxed\{\\text\{([A-Za-z])\}\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()

    # Fallback: single letter
    stripped = response.strip()
    if len(stripped) == 1 and stripped.upper() in OPTION_LETTERS:
        return stripped.upper()
    if len(stripped) == 2 and stripped[0].upper() in OPTION_LETTERS and stripped[1] in '.,':
        return stripped[0].upper()

    return None


def compute_swap_accuracy(results):
    """Compute accuracy breakdown by num_of_swaps from the results list."""
    from collections import defaultdict
    swap_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        swaps = r.get("num_of_swaps", "unknown")
        swap_stats[swaps]["total"] += 1
        if r.get("correct"):
            swap_stats[swaps]["correct"] += 1
    # Build summary dict
    summary = {}
    for swaps in sorted(swap_stats.keys()):
        s = swap_stats[swaps]
        summary[swaps] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
        }
    return summary


def save_results(results, output_path):
    """Save results + per-swap accuracy summary to file — called after each item."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_correct = sum(1 for r in results if r.get("correct"))
    total_items = len(results)
    swap_accuracy = compute_swap_accuracy(results)

    output = {
        "model": MODEL_NAME,
        "total": total_items,
        "correct": total_correct,
        "accuracy": round(total_correct / total_items * 100, 2) if total_items > 0 else 0,
        "accuracy_by_num_of_swaps": swap_accuracy,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    metadata_path = os.path.join(script_dir, "cup_games_filtered_metadata.json")
    video_dir = os.path.join(script_dir, "cup_games_filtered")
    output_path = os.path.join(script_dir, f"{MODEL_NAME}_cup_games_results.json")

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} videos from metadata")
    print(f"Video directory: {video_dir}")
    print(f"Output path: {output_path}")

    # Resume support: load existing results if present
    results = []
    processed_ids = set()
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {output_path} contains invalid JSON, starting fresh")
            saved = []
        # Handle both old flat list format and new nested format
        if isinstance(saved, list):
            results = saved
        elif isinstance(saved, dict) and "results" in saved:
            results = saved["results"]
        processed_ids = {r["video_id"] for r in results}
        print(f"Resuming: {len(processed_ids)} already processed")

    # Redo N/A entries: re-run videos where predicted answer is None
    na_indices = [i for i, r in enumerate(results) if r.get("predicted") is None]
    if na_indices:
        print(f"Found {len(na_indices)} N/A entries to redo.")
        for count, idx in enumerate(na_indices, 1):
            r = results[idx]
            video_id = r["video_id"]
            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            if not os.path.exists(video_path):
                print(f"[Redo {count}/{len(na_indices)}] SKIP {video_id}: video not found")
                continue

            prompt = build_prompt(r["question"], r["options"])

            print(f"\n{'='*60}")
            print(f"[Redo {count}/{len(na_indices)}] {video_id}")
            print(f"Old response (truncated): {r.get('response', '')[:120]}...")
            print(f"{'='*60}")

            try:
                response = analyze_video(video_path, prompt)
                print(f"Response: {response}")
            except Exception as e:
                response = f"Error: {e}"
                print(f"Error: {e}")

            predicted = extract_answer(response)
            is_correct = predicted == r["ground_truth_letter"] if predicted else False

            results[idx]["response"] = response
            results[idx]["predicted"] = predicted
            results[idx]["correct"] = is_correct

            save_results(results, output_path)
            print(f"Predicted: {predicted} | Correct: {is_correct}")
            print(f"Progress saved (redone {count}/{len(na_indices)})")

        print(f"\nRedo complete. {len(na_indices)} entries re-processed.")

    total = len(metadata)
    correct = sum(1 for r in results if r.get("correct"))

    for idx, (video_id, entry) in enumerate(metadata.items()):
        if video_id in processed_ids:
            continue

        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[{idx+1}/{total}] SKIP {video_id}: video not found")
            continue

        mc_questions = entry.get("mc_question", [])
        if not mc_questions:
            print(f"[{idx+1}/{total}] SKIP {video_id}: no questions")
            continue

        for q in mc_questions:
            question_text = q["question"]
            options = q["options"]
            answer_idx = q["answer"]  # 0-based index
            ground_truth_letter = OPTION_LETTERS[answer_idx]

            prompt = build_prompt(question_text, options)

            print(f"\n{'='*60}")
            print(f"[{idx+1}/{total}] {video_id}")
            print(f"Question: {question_text}")
            print(f"Options: {options}")
            print(f"Ground truth: {ground_truth_letter} ({options[answer_idx]})")
            print(f"{'='*60}")

            try:
                response = analyze_video(video_path, prompt)
                print(f"Response: {response}")
            except Exception as e:
                response = f"Error: {e}"
                print(f"Error: {e}")

            predicted = extract_answer(response)
            is_correct = predicted == ground_truth_letter if predicted else False

            if is_correct:
                correct += 1

            result = {
                "video_id": video_id,
                "question_id": q.get("id"),
                "question": question_text,
                "options": options,
                "ground_truth_index": answer_idx,
                "ground_truth_letter": ground_truth_letter,
                "model": MODEL_NAME,
                "response": response,
                "predicted": predicted,
                "correct": is_correct,
                "num_of_swaps": entry["metadata"].get("num_of_swaps"),
                "num_of_cups": entry["metadata"].get("num_of_cups"),
            }
            results.append(result)
            processed_ids.add(video_id)

            # Save after each item
            save_results(results, output_path)

            done = len(results)
            acc = correct / done * 100 if done > 0 else 0
            print(f"Predicted: {predicted} | Correct: {is_correct}")
            print(f"Running accuracy: {correct}/{done} = {acc:.2f}%")

            # Print running per-swap accuracy
            swap_acc = compute_swap_accuracy(results)
            swap_summary = " | ".join(
                f"{k} swaps: {v['correct']}/{v['total']}({v['accuracy']:.1f}%)"
                for k, v in sorted(swap_acc.items())
            )
            print(f"By swaps: {swap_summary}")

    # ── Final accuracy report ──
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS  —  {MODEL_NAME}")
    print(f"{'='*60}")

    total_items = len(results)
    total_correct = sum(1 for r in results if r.get("correct"))
    total_acc = total_correct / total_items * 100 if total_items > 0 else 0
    print(f"Overall accuracy: {total_correct}/{total_items} = {total_acc:.2f}%")

    # Accuracy by number of swaps
    swap_accuracy = compute_swap_accuracy(results)
    print(f"\nAccuracy by num_of_swaps:")
    print("-" * 40)
    for swaps, s in sorted(swap_accuracy.items()):
        print(f"  {swaps} swaps: {s['correct']}/{s['total']} = {s['accuracy']:.2f}%")

    # Save final results (includes accuracy_by_num_of_swaps in the JSON)
    save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
