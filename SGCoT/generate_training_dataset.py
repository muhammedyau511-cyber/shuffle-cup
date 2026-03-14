"""
Convert cup and card tracking trajectories into a single chat-style JSONL
training dataset (mixed).

Inputs:  trajectory_cup.json, trajectory_card.json
Output:  train.jsonl (combined, shuffled)

Each sample becomes a {"messages": [user, assistant]} entry with varied
question/answer templates (MCQ, direct, box).
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ─── Configuration ──────────────────────────────────────────────────────────
INPUT_CUP = "trajectory_cup.json"
INPUT_CARD = "trajectory_card.json"
OUTPUT_FILE = "train.jsonl"
SEED = 2026

OBJECT_NAME_CUP = "the cup that contains the ball"
OBJECT_NAME_CARD = "the Queen of Hearts"

# Thresholds to classify x-coordinate into left / middle / right
TH_LEFT = 333
TH_RIGHT = 666

# ─── Question / Answer Templates ───────────────────────────────────────────
TEMPLATES_MCQ_CUP = [
    {
        "question": "Track {obj} and answer which cup contains the ball at the end of the video.",
        "answer": "{tracks} Answer: {pos}.",
    }
]

TEMPLATES_MCQ_CARD = [
    {
        "question": "Track {obj} and answer where is the Queen of Hearts at the end of the video.",
        "answer": "{tracks} Answer: {pos}.",
    }
]

# TEMPLATES_MCQ_CUP = [
#     {
#         "question": "Track {obj}, and answer: Which cup contains the ball at the end of the video?",
#         "answer": "{tracks} Answer: {pos}.",
#     }
# ]

# TEMPLATES_MCQ_CARD = [
#     {
#         "question": "Track {obj}, and answer: Where is the Queen of Hearts at the end of the video?",
#         "answer": "{tracks} Answer: {pos}.",
#     }
# ]

ALL_TEMPLATE_GROUPS_CUP = [("mcq", TEMPLATES_MCQ_CUP)]
ALL_TEMPLATE_GROUPS_CARD = [("mcq", TEMPLATES_MCQ_CARD)]

# ─── Helpers ────────────────────────────────────────────────────────────────

def parse_final_position(final_answer: str) -> str:
    """Extract position name from 'A. left' / 'B. middle' / 'C. right'."""
    # return final_answer.split(". ", 1)[1]
    return final_answer


def pos_to_letter(pos: str) -> str:
    return {"left": "A", "middle": "B", "right": "C"}[pos]


def build_sample(
    entry: Dict[str, Any],
    object_name: str,
    all_template_groups: List[Tuple[str, List[Dict[str, str]]]],
) -> Dict[str, Any]:
    """Convert one tracking entry to a chat messages sample."""
    track_output = entry["track_output"]
    pos = parse_final_position(entry["final_answer"])
    letter = pos_to_letter(pos)
    pos_cap = pos.capitalize()

    group_name, templates = random.choice(all_template_groups)
    tmpl = random.choice(templates)

    question = tmpl["question"].format(obj=object_name)
    answer = tmpl["answer"].format(
        tracks=track_output,
        obj=object_name,
        pos=pos,
        letter=letter,
        pos_cap=pos_cap,
    )

    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }



def main():
    random.seed(SEED)
    script_dir = Path(__file__).resolve().parent

    # Load cup and card tracking data
    with open(script_dir / INPUT_CUP, "r", encoding="utf-8") as f:
        cup_data: List[Dict[str, Any]] = json.load(f)
    with open(script_dir / INPUT_CARD, "r", encoding="utf-8") as f:
        card_data: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(cup_data)} entries from {INPUT_CUP}")
    print(f"Loaded {len(card_data)} entries from {INPUT_CARD}")

    # Convert each source to samples
    cup_samples = [
        build_sample(entry, OBJECT_NAME_CUP, ALL_TEMPLATE_GROUPS_CUP)
        for entry in cup_data
    ]
    card_samples = [
        build_sample(entry, OBJECT_NAME_CARD, ALL_TEMPLATE_GROUPS_CARD)
        for entry in card_data
    ]

    # Combine and mix (shuffle)
    samples = cup_samples + card_samples
    random.shuffle(samples)

    # Write combined JSONL
    out_path = script_dir / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    print(f"✅  Wrote {len(samples)} samples → {OUTPUT_FILE}")
    print(f"    Cup: {len(cup_samples)}, Card: {len(card_samples)} (mixed)")
    print()

    # Show a few examples
    for i in [0, 1, 2]:
        s = samples[i]
        q = s["messages"][0]["content"][0]["text"]
        a = s["messages"][1]["content"][0]["text"]
        print(f"--- Sample {i+1} ---")
        print(f"  Q: {q[:120]}{'...' if len(q) > 120 else ''}")
        print(f"  A: {a[:120]}{'...' if len(a) > 120 else ''}")
        print()


if __name__ == "__main__":
    main()
