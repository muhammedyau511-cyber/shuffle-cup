"""
Generate synthetic cup-tracking trajectories from movement primitives
extracted from real data in real_cup.json.

Trajectory format: <tracks coords="t obj x y; ...">
  - 25 frames, t=0.0 to 12.0, step 0.5s
  - obj is always 1
  - x, y in [0, 1000]

Cup positions: left (~218,520), middle (~504,525), right (~775,525)
"""

import json
import math
import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REAL_FILE = "real_data/real_cup.json"
OUTPUT_FILE = "trajectory_cup.json"

NUM_SAMPLES = 200
SEED = 42

DT = 0.5
TOTAL_FRAMES = 25  # indices 0-24

LEFT_MAX_X = 350
RIGHT_MIN_X = 650

REST_COORDS: Dict[str, Tuple[int, int]] = {
    "left":   (218, 520),
    "middle": (504, 525),
    "right":  (775, 525),
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


def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Template extraction from real data
# ---------------------------------------------------------------------------
def extract_primitives(real_data: list):
    """
    Walk each real trajectory and extract:
      lifts    {pos: [4-frame list, ...]}
      swaps    {(src,dst): [4-frame arc, ...]}     arc = 3 inner + 1 arrival
    Feints are no longer extracted from real data (they contain teleport jumps);
    smooth feints are generated dynamically from swap arcs instead.
    """
    lifts:  Dict[str, List[List[Tuple[int, int]]]] = {}
    swaps:  Dict[Tuple[str, str], List[List[Tuple[int, int]]]] = {}

    for entry in real_data:
        frames = parse_frames(entry["track_output"])
        if len(frames) != TOTAL_FRAMES:
            continue

        start_pos = classify_x(frames[0][0])
        lifts.setdefault(start_pos, []).append(list(frames[0:4]))

        for blk in (4, 8, 12, 16, 20):
            blk_end = blk + 4
            if blk_end >= TOTAL_FRAMES:
                break

            departure = frames[blk]
            arrival   = frames[blk_end]
            src = classify_x(departure[0])
            dst = classify_x(arrival[0])

            if src == dst:
                continue

            action = list(frames[blk + 1 : blk_end + 1])  # 4 frames
            full_block = [departure] + action
            if all(
                euclidean(full_block[i], full_block[i + 1]) < 400
                for i in range(len(full_block) - 1)
            ):
                swaps.setdefault((src, dst), []).append(action)

    return lifts, swaps


# ---------------------------------------------------------------------------
# Hardcoded fallback templates (derived from manual analysis of real data)
# ---------------------------------------------------------------------------
FALLBACK_SWAPS: Dict[Tuple[str, str], List[List[Tuple[int, int]]]] = {
    ("left", "right"): [
        [(351, 342), (542, 295), (712, 378), (775, 517)],
        [(321, 764), (549, 822), (745, 685), (775, 525)],
    ],
    ("right", "left"): [
        [(647, 330), (458, 289), (294, 378), (228, 500)],
        [(682, 772), (445, 822), (256, 678), (217, 522)],
    ],
    ("left", "middle"): [
        [(265, 672), (369, 702), (474, 624), (502, 528)],
        [(295, 396), (393, 372), (477, 428), (504, 522)],
    ],
    ("middle", "left"): [
        [(447, 402), (349, 378), (260, 431), (218, 522)],
        [(440, 672), (329, 703), (235, 633), (222, 524)],
    ],
    ("middle", "right"): [
        [(554, 396), (643, 372), (742, 422), (774, 525)],
        [(564, 672), (679, 711), (767, 622), (774, 528)],
    ],
    ("right", "middle"): [
        [(707, 389), (608, 369), (527, 424), (504, 530)],
        [(738, 672), (629, 707), (528, 628), (504, 530)],
    ],
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def make_smooth_feint(
    src_pos: str,
    swaps: Dict[Tuple[str, str], List[List[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """
    Build a smooth 4-frame feint: cup moves partway toward a neighbour
    then smoothly returns.  No teleport jumps.

    Arc shape:  rest -> quarter -> peak -> quarter -> rest
    The "peak" is the first frame of a randomly-chosen swap arc from src,
    and "quarter" is the midpoint between rest and peak.
    """
    src_rest = REST_COORDS[src_pos]
    targets = [d for (s, d) in swaps if s == src_pos]
    if not targets:
        rx, ry = src_rest
        return [(rx, ry - 30), (rx, ry - 50), (rx, ry - 30), (rx, ry)]

    dst = random.choice(targets)
    arc = random.choice(swaps[(src_pos, dst)])
    peak = arc[0]  # first intermediate frame of a real swap

    qx = (src_rest[0] + peak[0]) // 2
    qy = (src_rest[1] + peak[1]) // 2
    quarter = (qx, qy)

    return [quarter, peak, quarter, src_rest]
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
    lifts: dict,
    swaps: dict,
) -> Tuple[List[Tuple[int, int]], str]:
    """Build a 25-frame trajectory.  Returns (frames, final_position)."""
    traj: List[Tuple[int, int]] = []

    # --- Lift (frames 0-3) ---
    if lifts.get(start_pos):
        traj.extend(jitter(random.choice(lifts[start_pos]), 3))
    else:
        rx, ry = REST_COORDS[start_pos]
        traj.extend([
            (rx, ry),
            (rx + random.randint(-10, 10), 307),
            (rx + random.randint(-10, 10), 307),
            (rx + random.randint(-5, 5), 400 + random.randint(-10, 10)),
        ])

    # --- Frame 4: departure at rest position ---
    cur_pos = start_pos
    rx, ry = REST_COORDS[cur_pos]
    traj.append((rx + random.randint(-5, 5), ry + random.randint(-10, 30)))

    # --- 5 action blocks (each 4 frames) ---
    swap_count = 0

    for blk_idx in range(5):
        remaining = 5 - blk_idx
        can_swap = [(s, d) for (s, d) in swaps if s == cur_pos]

        # Force at least one swap
        if swap_count == 0 and remaining <= 1:
            action = "swap" if can_swap else "rest"
        else:
            r = random.random()
            if r < 0.55 and can_swap:
                action = "swap"
            else:
                action = "rest"

        if action == "swap" and can_swap:
            src, dst = random.choice(can_swap)
            arc = random.choice(swaps[(src, dst)])
            traj.extend(jitter(arc, 5))
            cur_pos = dst
            swap_count += 1

        else:
            traj.extend(rest_frames(cur_pos))

    # Guarantee exactly 25 frames
    traj = traj[:TOTAL_FRAMES]
    while len(traj) < TOTAL_FRAMES:
        traj.append(traj[-1])

    return traj, cur_pos


def frames_to_coords(frames: List[Tuple[int, int]]) -> str:
    return ";".join(f"{i * DT:.1f} 1 {x} {y}" for i, (x, y) in enumerate(frames))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)

    with open(REAL_FILE, encoding="utf-8") as f:
        real_data = json.load(f)

    lifts, swaps = extract_primitives(real_data)

    # Merge fallbacks into extracted templates
    for key, arcs in FALLBACK_SWAPS.items():
        swaps.setdefault(key, []).extend(arcs)

    # --- Generate ---
    samples = []
    answer_map = {"left": "left", "middle": "middle", "right": "right"}

    for i in range(NUM_SAMPLES):
        start_pos = random.choice(POSITIONS)
        frames, final_pos = generate_trajectory(start_pos, lifts, swaps)

        coords_str = frames_to_coords(frames)

        track_tag = (
            f'<tracks coords="{coords_str}">'
            f"the cup that contains the ball</tracks>"
        )

        track_output = f"{track_tag}"

        samples.append({
            "file_id": f"cup_{i + 28:03d}.mp4",
            "track_output": track_output,
            "final_answer": answer_map[final_pos],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # --- Stats ---
    start_dist = {p: 0 for p in POSITIONS}
    end_dist   = {p: 0 for p in POSITIONS}
    for s in samples:
        fs = parse_frames(s["track_output"])
        start_dist[classify_x(fs[0][0])] += 1
        end_dist[classify_x(fs[-1][0])] += 1

    print(f"Generated {len(samples)} samples -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
