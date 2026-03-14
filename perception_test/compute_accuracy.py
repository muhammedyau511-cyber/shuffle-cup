#!/usr/bin/env python3
"""
Compute accuracy for all cup_games_results JSON files in the perception_test directory.
Recomputes statistics from the results array and outputs in the same format.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def compute_swap_accuracy(results, num_of_cups=None):
    """Compute accuracy breakdown by num_of_swaps. If num_of_cups is set, only include those results."""
    swap_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if num_of_cups is not None and r.get("num_of_cups") != num_of_cups:
            continue
        swaps = r.get("num_of_swaps", "unknown")
        swap_stats[swaps]["total"] += 1
        if r.get("correct"):
            swap_stats[swaps]["correct"] += 1
    summary = {}
    for swaps in sorted(swap_stats.keys(), key=lambda x: (isinstance(x, str), x)):
        s = swap_stats[swaps]
        summary[str(swaps)] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
        }
    return summary


def compute_cups_accuracy(results):
    """Compute accuracy breakdown by num_of_cups."""
    cup_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cups = r.get("num_of_cups", "unknown")
        cup_stats[cups]["total"] += 1
        if r.get("correct"):
            cup_stats[cups]["correct"] += 1
    summary = {}
    for cups in sorted(cup_stats.keys(), key=lambda x: (isinstance(x, str), x)):
        s = cup_stats[cups]
        summary[str(cups)] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
        }
    return summary


def compute_3cups_swapped_accuracy(results):
    """Compute accuracy for cases where num_of_cups == 3 and num_of_swaps > 0."""
    filtered = [r for r in results if r.get("num_of_cups") == 3 and (r.get("num_of_swaps") or 0) > 0]
    if not filtered:
        return {"correct": 0, "total": 0, "accuracy": 0}
    correct = sum(1 for r in filtered if r.get("correct"))
    total = len(filtered)
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
    }


def process_file(json_path):
    """Process a single results JSON file and recompute accuracy."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return None

    model = data.get("model", json_path.stem.replace("_cup_games_results", ""))
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = round(correct / total * 100, 2) if total > 0 else 0

    output = {
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_by_num_of_swaps": compute_swap_accuracy(results, num_of_cups=3),
        "accuracy_by_num_of_cups": compute_cups_accuracy(results),
        "accuracy_3cups_swapped": compute_3cups_swapped_accuracy(results),
        "results": results,
    }
    return output


def print_summary(data, filepath):
    """Print a formatted summary for one file."""
    name = filepath.name
    model = data["model"]
    total = data["total"]
    correct = data["correct"]
    accuracy = data["accuracy"]

    print(f"\n{'='*70}")
    print(f"  File  : {name}")
    print(f"  Model : {model}")
    print(f"  Overall: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'='*70}")

    print(f"\n  Accuracy by num_of_swaps (num_of_cups=3 only):")
    print(f"  {'Swaps':>6}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}")
    print(f"  {'-'*35}")
    for swaps, s in data["accuracy_by_num_of_swaps"].items():
        print(f"  {swaps:>6}  {s['correct']:>8}  {s['total']:>6}  {s['accuracy']:>8.2f}%")

    print(f"\n  Accuracy by num_of_cups:")
    print(f"  {'Cups':>6}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}")
    print(f"  {'-'*35}")
    for cups, s in data["accuracy_by_num_of_cups"].items():
        print(f"  {cups:>6}  {s['correct']:>8}  {s['total']:>6}  {s['accuracy']:>8.2f}%")

    s = data["accuracy_3cups_swapped"]
    print(f"\n  3 cups & swaps > 0: {s['correct']}/{s['total']} = {s['accuracy']:.2f}%")

    failed = [r for r in data["results"] if r.get("predicted") is None]
    if failed:
        print(f"\n  Extraction failures ({len(failed)}):")
        for r in failed:
            resp = r.get("response", "")
            preview = resp[:80].replace("\n", " ")
            if len(resp) > 80:
                preview += "..."
            print(f"    - {r['video_id']} (GT: {r.get('ground_truth_letter', '?')}): {preview}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute accuracy for cup_games_results JSON files"
    )
    parser.add_argument(
        "files", nargs="*",
        help="JSON files to process (default: all *_cup_games_results*.json in this directory)"
    )
    parser.add_argument(
        "-w", "--write", action="store_true",
        help="Overwrite each JSON file with recomputed accuracy stats"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        json_files = sorted(script_dir.glob("*_cup_games_results*.json"))

    if not json_files:
        print("No result JSON files found.")
        sys.exit(1)

    all_data = []

    for jf in json_files:
        if not jf.exists():
            print(f"Warning: {jf} not found, skipping")
            continue

        data = process_file(jf)
        if data is None:
            print(f"Warning: {jf} has no results, skipping")
            continue

        print_summary(data, jf)
        all_data.append((jf, data))

        if args.write:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  >> Updated {jf.name}")

    if len(all_data) > 1:
        print(f"\n{'='*70}")
        print("  SUMMARY ACROSS ALL FILES")
        print(f"{'='*70}")
        print(f"  {'Model':<45} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
        print(f"  {'-'*72}")
        for jf, d in all_data:
            label = d["model"]
            suffix = ""
            stem = jf.stem
            base = f"{d['model']}_cup_games_results"
            if stem != base:
                suffix = stem.replace(base, "")
            if suffix:
                label = f"{d['model']}{suffix}"
            print(f"  {label:<45} {d['correct']:>8} {d['total']:>6} {d['accuracy']:>8.2f}%")


if __name__ == "__main__":
    main()
