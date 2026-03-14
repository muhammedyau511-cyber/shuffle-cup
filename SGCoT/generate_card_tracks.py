"""
Generate synthetic card-tracking trajectories from movement primitives
extracted from real data in batch_results_sot_card_27_27.json.

Trajectory format: <tracks coords="t obj x y; ...">
  - 25 frames, t=0.0 to 12.0, step 0.5s
  - obj is always 1
  - x, y in [0, 1000]

Card positions: left (~250,~500), middle (~500,~500), right (~750,~500)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REAL_DATA_PATH = "real_data/real_card.json"
OUTPUT_FILE = "trajectory_card.json"

NUM_SAMPLES = 100
SEED = 2025

DT = 0.5
TOTAL_FRAMES = 25  # indices 0-24

LEFT_MAX_X = 370
RIGHT_MIN_X = 630

REST_COORDS: Dict[str, Tuple[int, int]] = {
    "left":   (250, 500),
    "middle": (500, 500),
    "right":  (750, 500),
}

# Initial-frame y-offsets (only for the very first frame of the video)
INIT_Y: Dict[str, int] = {
    "left":   497,
    "middle": 504,
    "right":  517,
}

POSITIONS = ["left", "middle", "right"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def classify_x(x: int) -> str:
    if x < LEFT_MAX_X:
        return "left"
    if x > RIGHT_MIN_X:
        return "right"
    return "middle"


def parse_frames(track_output: str) -> List[Tuple[int, int]]:
    coords_str = track_output.split('coords="')[1].split('"')[0]
    frames = []
    for part in coords_str.split(";"):
        tokens = part.strip().split()
        if len(tokens) >= 4:
            frames.append((int(tokens[2]), int(tokens[3])))
    return frames


# ---------------------------------------------------------------------------
# Template extraction from real data
# ---------------------------------------------------------------------------
def extract_primitives(real_data: list):
    lifts_rest: Dict[str, List[List[Tuple[int, int]]]] = {}
    lifts_swap: Dict[str, List[List[Tuple[int, int]]]] = {}
    swaps: Dict[Tuple[str, str], List[List[Tuple[int, int]]]] = {}

    for entry in real_data:
        frames = parse_frames(entry["track_output"])
        if len(frames) != TOTAL_FRAMES:
            continue

        boundary_indices = [3, 7, 11, 15, 19, 23]
        positions = [classify_x(frames[f][0]) for f in boundary_indices]

        start_pos = positions[0]
        lift = list(frames[0:4])
        f3_y_displaced = abs(lift[3][1] - 500) > 30
        if f3_y_displaced:
            lifts_swap.setdefault(start_pos, []).append(lift)
        else:
            lifts_rest.setdefault(start_pos, []).append(lift)

        for blk in range(5):
            src = positions[blk]
            dst = positions[blk + 1]
            if src != dst:
                blk_start = 4 + blk * 4
                action_frames = list(frames[blk_start : blk_start + 4])
                swaps.setdefault((src, dst), []).append(action_frames)

    return lifts_rest, lifts_swap, swaps


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def jitter(coords: List[Tuple[int, int]], amt: int = 5) -> List[Tuple[int, int]]:
    return [
        (
            max(0, min(1000, x + random.randint(-amt, amt))),
            max(0, min(1000, y + random.randint(-amt, amt))),
        )
        for x, y in coords
    ]


def rest_frames(pos: str, n: int = 4, noise: int = 3) -> List[Tuple[int, int]]:
    rx, ry = REST_COORDS[pos]
    return [
        (rx + random.randint(-noise, noise), ry + random.randint(-noise, noise))
        for _ in range(n)
    ]


def generate_trajectory(
    start_pos: str,
    lifts_rest: dict,
    lifts_swap: dict,
    swaps: dict,
) -> Tuple[List[Tuple[int, int]], str]:
    """Build a 25-frame trajectory. Returns (frames, final_position)."""
    traj: List[Tuple[int, int]] = []

    # Pre-plan: decide all 5 block actions first so we can pick the right lift
    cur_pos = start_pos
    swap_count = 0
    block_plan: List[Tuple[str, str, str]] = []  # (action_type, src, dst)

    for blk_idx in range(5):
        remaining = 5 - blk_idx
        can_swap = [(s, d) for (s, d) in swaps if s == cur_pos and d != cur_pos]

        if swap_count == 0 and remaining <= 1:
            action = "swap" if can_swap else "rest"
        else:
            r = random.random()
            action = "swap" if (r < 0.55 and can_swap) else "rest"

        if action == "swap" and can_swap:
            src, dst = random.choice(can_swap)
            block_plan.append(("swap", src, dst))
            cur_pos = dst
            swap_count += 1
        else:
            block_plan.append(("rest", cur_pos, cur_pos))

    final_pos = cur_pos

    # --- Initial block (frames 0-3) ---
    # Pick lift type based on whether block 1 is a swap or rest
    block1_is_swap = block_plan[0][0] == "swap"
    if block1_is_swap and lifts_swap.get(start_pos):
        traj.extend(jitter(random.choice(lifts_swap[start_pos]), 3))
    elif not block1_is_swap and lifts_rest.get(start_pos):
        traj.extend(jitter(random.choice(lifts_rest[start_pos]), 3))
    elif lifts_rest.get(start_pos):
        traj.extend(jitter(random.choice(lifts_rest[start_pos]), 3))
    elif lifts_swap.get(start_pos):
        # Fallback: use swap lift but clamp f3 y to near 500
        lift = list(random.choice(lifts_swap[start_pos]))
        lift[3] = (lift[3][0], 500 + random.randint(-5, 5))
        traj.extend(jitter(lift, 3))
    else:
        traj.extend(rest_frames(start_pos))

    # --- 5 action blocks (each 4 frames) ---
    for blk_i, (act_type, src, dst) in enumerate(block_plan):
        if act_type == "swap":
            block = [p for p in random.choice(swaps[(src, dst)])]
            # If next block is rest, clamp arrival y near 500 (matches real data
            # where swap→rest transitions have ≤20 dy at boundary).
            next_is_rest = (
                blk_i + 1 >= len(block_plan)
                or block_plan[blk_i + 1][0] == "rest"
            )
            if next_is_rest:
                x, y = block[-1]
                ry = REST_COORDS[dst][1]
                block[-1] = (x, ry + random.randint(-8, 8))
            traj.extend(jitter(block, 5))
        else:
            traj.extend(rest_frames(src))

    # --- Trailing frame (f24) ---
    rx, ry = REST_COORDS[final_pos]
    traj.append((rx + random.randint(-5, 5), ry + random.randint(-5, 5)))

    traj = traj[:TOTAL_FRAMES]
    while len(traj) < TOTAL_FRAMES:
        traj.append(traj[-1])

    return traj, final_pos


def frames_to_coords(frames: List[Tuple[int, int]]) -> str:
    return ";".join(f"{i * DT:.1f} 1 {x} {y}" for i, (x, y) in enumerate(frames))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)

    with open(REAL_DATA_PATH, encoding="utf-8") as f:
        real_data = json.load(f)

    lifts_rest, lifts_swap, swaps = extract_primitives(real_data)

    answer_map = {"left": "left", "middle": "middle", "right": "right"}

    samples = []
    for i in range(NUM_SAMPLES):
        start_pos = random.choice(POSITIONS)
        frames, final_pos = generate_trajectory(
            start_pos, lifts_rest, lifts_swap, swaps
        )
        coords_str = frames_to_coords(frames)

        track_tag = (
            f'<tracks coords="{coords_str}">'
            f"the Queen of Hearts</tracks>"
        )

        samples.append({
            "file_id": f"card_{i + 1:03d}.mp4",
            "track_output": track_tag,
            "final_answer": answer_map[final_pos],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # --- Stats ---
    start_dist = {p: 0 for p in POSITIONS}
    end_dist = {p: 0 for p in POSITIONS}
    for s in samples:
        fs = parse_frames(s["track_output"])
        start_dist[classify_x(fs[0][0])] += 1
        end_dist[classify_x(fs[-1][0])] += 1

    print(f"Generated {len(samples)} samples -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
